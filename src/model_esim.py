import tensorflow as tf
import numpy as np

import numpy as np
from keras.layers import *
from keras.activations import softmax
from keras.models import Model

def biRNN(inputs, length, hidden_size, training, dropout_rate=0.1, name="biRNN", reuse=False): 
  with tf.variable_scope(name, reuse=reuse):
    cell = tf.nn.rnn_cell.BasicLSTMCell
    fw_cell = cell(hidden_size, name='fw')
    bw_cell = cell(hidden_size, name='bw')
    if training:
      fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=(1 - dropout_rate))
      bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=(1 - dropout_rate))
    
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                  bw_cell,
                                                  inputs,
                                                  sequence_length=length,
                                                  dtype=tf.float32)
    output = tf.concat([outputs[0], outputs[1]], axis=2)
  return output

def aggregate(input_1, input_2, num_dense=200, dropout_rate=0.5):
    feat1 = concatenate([GlobalAvgPool1D()(input_1), GlobalMaxPool1D()(input_1)])
    feat2 = concatenate([GlobalAvgPool1D()(input_2), GlobalMaxPool1D()(input_2)])

    x = concatenate([feat1, feat2])
    x = Dropout(dropout_rate)(x)
    
    x = Dense(num_dense, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    return x    

def align(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

def proj(x, unit, dropout_rate):
  x = Dense(unit, activation='relu')(x)
  x = Dropout(dropout_rate)(x)
  return x

class ModelESIM(object):
  def __init__(self, params, word2vec, features, labels, training=False):
    len1, len2, s1, s2 = features
    embed_dim     = params['embed_dim']
    hidden_size   = params['hidden_size'] # 300
    input_keep    = 0.8
    learning_rate = 0.0005
    max_norm      = 10
    l2_coef       = 0.0001

    K.set_learning_phase(training)
    
    with tf.device('/cpu:0'):
      embedding = tf.get_variable("word2vec", initializer=word2vec, trainable=False)
      s1 = tf.nn.embedding_lookup(embedding, s1)
      s2 = tf.nn.embedding_lookup(embedding, s2)
    if training:
      s1 = tf.nn.dropout(s1, input_keep)
      s2 = tf.nn.dropout(s2, input_keep)

    # Encoding
    q1_encoded = biRNN(s1, len1, hidden_size, training, 0.1, "encode")
    q2_encoded = biRNN(s2, len2, hidden_size, training, 0.1, "encode", reuse=True)
    
    # Alignment
    q1_aligned, q2_aligned = align(q1_encoded, q2_encoded)
    
    # Compare
    q1_combined = concatenate([q1_encoded, q2_aligned, q1_encoded-q2_aligned, q1_encoded, q2_aligned])
    q2_combined = concatenate([q2_encoded, q1_aligned, q2_encoded-q1_aligned, q2_encoded, q1_aligned])

    q1_proj = proj(q1_combined, hidden_size, 0.5)
    q2_proj = proj(q2_combined, hidden_size, 0.5)

    q1_compare = biRNN(q1_proj, len1, hidden_size, training, 0.1, "compare")
    q2_compare = biRNN(q2_proj, len2, hidden_size, training, 0.1, "compare", reuse=True)
    
    # Aggregate
    x = aggregate(q1_compare, q2_compare)
    
    logits = tf.squeeze(Dense(1)(x))

    self.prob = tf.sigmoid(logits)
    self.pred = tf.rint(self.prob)
    self.acc = tf.metrics.accuracy(labels=labels, predictions=self.pred)

    self.loss = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=tf.to_float(labels), logits=logits))
    l2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                    if 'bias' not in v.name ]) * l2_coef
    self.loss += l2

    if training:
      self.global_step = tf.train.get_or_create_global_step()
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, max_norm)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables), 
                                                  global_step=self.global_step)
        # self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
