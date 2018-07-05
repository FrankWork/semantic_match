import tensorflow as tf
import numpy as np

import numpy as np
from keras.layers import *
from keras.activations import softmax

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

def highway(x, units, scope, training, reuse=False):
  # in_val: [batch_size, passage_len, dim]
  with tf.variable_scope(scope, reuse=reuse):
    if x.shape.as_list()[-1] != units:
      x = tf.layers.dense(x, units, activation=tf.nn.relu)

    gate = tf.layers.dense(x, units, activation=tf.nn.sigmoid)
    h    = tf.layers.dense(x, units, activation=tf.nn.relu)
    y = h*gate + (1-gate)*x
    if training:
      y = tf.nn.dropout(y, 0.8)
    return y

def aggregate(input_1, input_2, units, training, dropout_rate=0.5):
    p = tf.concat([tf.reduce_mean(input_1, axis=1), 
                   tf.reduce_max(input_1, axis=1)],
                   axis=-1)
    q = tf.concat([tf.reduce_mean(input_2, axis=1), 
                   tf.reduce_max(input_2, axis=1)], 
                   axis=-1)

    x = tf.concat([p, q, tf.abs(p-q), p*q], axis=-1)

    if training:
      x = tf.nn.dropout(x, 1-dropout_rate)
    # x = Dense(units, activation='relu')(x)
    # x = Dropout(dropout_rate)(x)

    x = highway(x, units, 'h1', training)
    x = highway(x, units, 'h2', training)
    return x   


class ModelESIM(object):
  def __init__(self, params, word2vec, features, labels, training=False):
    len1, len2, s1, s2 = features
    embed_dim     = params['embed_dim']
    hidden_size   = 200
    input_keep    = 0.8
    learning_rate = 1e-3
    max_norm      = 10
    l2_coef       = 1e-5

    K.set_learning_phase(training)
    
    with tf.device('/cpu:0'):
      embedding = tf.get_variable("word2vec", initializer=word2vec, trainable=False)
      s1 = tf.nn.embedding_lookup(embedding, s1)
      s2 = tf.nn.embedding_lookup(embedding, s2)
    if training:
      s1 = tf.nn.dropout(s1, input_keep)
      s2 = tf.nn.dropout(s2, input_keep)

    s1 = highway(s1, embed_dim, 'highway_in', training)
    s2 = highway(s2, embed_dim, 'highway_in', training, reuse=True)

    # Encoding
    q1_encoded = biRNN(s1, len1, hidden_size, training, 0.1, "encode")
    q2_encoded = biRNN(s2, len2, hidden_size, training, 0.1, "encode", reuse=True)
    
    # Alignment
    q1_aligned, q2_aligned = align(q1_encoded, q2_encoded)
    
    # Compare
    q1_combined = concatenate([q1_encoded, q2_aligned, tf.abs(q1_encoded-q2_aligned), q1_encoded*q2_aligned])
    q2_combined = concatenate([q2_encoded, q1_aligned, tf.abs(q2_encoded-q1_aligned), q2_encoded*q1_aligned])

    q1_proj = proj(q1_combined, hidden_size, 0.5)
    q2_proj = proj(q2_combined, hidden_size, 0.5)

    q1_compare = biRNN(q1_proj, len1, hidden_size, training, 0.1, "compare")
    q2_compare = biRNN(q2_proj, len2, hidden_size, training, 0.1, "compare", reuse=True)
    
    # Aggregate
    x = aggregate(q1_compare, q2_compare, hidden_size, training)
        
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
      learning_rate = tf.minimum(0.0005, 
          0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, max_norm)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables), 
                                                  global_step=self.global_step)
        # self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
