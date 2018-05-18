import tensorflow as tf
import numpy as np

import numpy as np
from keras.layers import *
from keras.activations import softmax
from keras.models import Model

def biRNN(inputs, length, hidden_size, name="biRNN", reuse=False):
  # rnn_cells = {
  #   'GRU': tf.nn.rnn_cell.GRUCell,
  #   'LSTM': tf.nn.rnn_cell.BasicLSTMCell,
  # }
  cell = tf.nn.rnn_cell.BasicLSTMCell
  with tf.variable_scope(name, reuse=reuse):
    fw_rnn_cell = cell(hidden_size)
    bw_rnn_cell = cell(hidden_size)
    
    rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell,
                                                        bw_rnn_cell,
                                                        inputs,
                                                        sequence_length=length,
                                                        dtype=tf.float32)
    output = tf.concat([rnn_outputs[0], rnn_outputs[1]], axis=2)
  return output

def subtract(input_1, input_2):
    minus_input_2 = Lambda(lambda x: -x)(input_2)
    return add([input_1, minus_input_2])

def aggregate(input_1, input_2, num_dense=300, dropout_rate=0.5):
    feat1 = concatenate([GlobalAvgPool1D()(input_1), GlobalMaxPool1D()(input_1)])
    feat2 = concatenate([GlobalAvgPool1D()(input_2), GlobalMaxPool1D()(input_2)])
    x = concatenate([feat1, feat2])
    x = BatchNormalization()(x)
    x = Dense(num_dense, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_dense, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return x    

def align(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

class ModelESIM(object):
  def __init__(self, params, word2vec, features, labels, training=False):
    len1, len2, s1, s2 = features
    embed_dim     = params['embed_dim']
    hidden_size   = params['hidden_size']
    dropout       = params['dropout']
    learning_rate = params['learning_rate']
    max_norm      = 10

    K.set_learning_phase(training)
    
    with tf.device('/cpu:0'):
      embedding = tf.get_variable("word2vec", initializer=word2vec, trainable=False)
      s1 = tf.nn.embedding_lookup(embedding, s1)
      s2 = tf.nn.embedding_lookup(embedding, s2)

    q1_embed = BatchNormalization(axis=2)(s1)
    q2_embed = BatchNormalization(axis=2)(s2)

    # Encoding
    # encode = Bidirectional(LSTM(hidden_size, return_sequences=True))
    # q1_encoded = encode(q1_embed)
    # q2_encoded = encode(q2_embed)
    q1_encoded = biRNN(q1_embed, len1, hidden_size, "encode")
    q2_encoded = biRNN(q2_embed, len2, hidden_size, "encode")
    
    # Alignment
    q1_aligned, q2_aligned = align(q1_encoded, q2_encoded)
    
    # Compare
    q1_combined = concatenate([q1_encoded, q2_aligned, subtract(q1_encoded, q2_aligned), multiply([q1_encoded, q2_aligned])])
    q2_combined = concatenate([q2_encoded, q1_aligned, subtract(q2_encoded, q1_aligned), multiply([q2_encoded, q1_aligned])]) 
    # compare = Bidirectional(LSTM(hidden_size, return_sequences=True))
    # q1_compare = compare(q1_combined)
    # q2_compare = compare(q2_combined)
    q1_compare = biRNN(q1_combined, len1, hidden_size, "compare")
    q2_compare = biRNN(q2_combined, len2, hidden_size, "compare")
    
    # Aggregate
    x = aggregate(q1_compare, q2_compare)
    
    logits = tf.squeeze(Dense(1)(x))

    self.prob = tf.sigmoid(logits)
    self.pred = tf.rint(self.prob)
    self.acc = tf.metrics.accuracy(labels=labels, predictions=self.pred)

    self.loss = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=tf.to_float(labels), logits=logits))

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
