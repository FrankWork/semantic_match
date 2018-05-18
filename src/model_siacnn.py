import tensorflow as tf
import numpy as np

L = tf.keras.layers
K = tf.keras.backend
TimeDistributed = L.TimeDistributed
Dense = L.Dense
Lambda = L.Lambda
concatenate = L.concatenate
Dropout = L.Dropout
BatchNormalization = L.BatchNormalization

def cnn_layer(embedded_x, num_filters, filter_size, reuse=False):
  embedding_dim = embedded_x.get_shape().as_list()[-1]
  embedded_x_expanded = tf.expand_dims(embedded_x, -1)
  with tf.variable_scope('convolution', reuse=reuse):
    convoluted = tf.layers.conv2d(embedded_x_expanded,
                                    filters=num_filters,
                                    kernel_size=[filter_size, embedding_dim],
                                    activation=tf.nn.relu)
  with tf.variable_scope('pooling', reuse=reuse):
    convoluted = tf.squeeze(convoluted)
    pooling = tf.reduce_max(convoluted, axis=1)
  return pooling

def cnn_layers(embedded_x, num_filters, filter_sizes, reuse=False):
  pooled_flats = []
  with tf.variable_scope('cnn_network', reuse=reuse):
    for i, (n, size) in enumerate(zip(num_filters, filter_sizes)):
      with tf.variable_scope('cnn_layer_{}'.format(i), reuse=reuse):
        pooled_flat = cnn_layer(embedded_x, num_filters=n, filter_size=size, reuse=reuse)
        pooled_flats.append(pooled_flat)
    cnn_output = tf.concat(pooled_flats, axis=1)
    cnn_output = tf.reshape(cnn_output, [-1, np.sum(num_filters)])
  return cnn_output

class ModelSiameseCNN(object):
  def __init__(self, params, word2vec, features, labels, training=False):
    len1, len2, s1, s2 = features
    embed_dim     = params['embed_dim']
    hidden_size   = params['hidden_size']
    dropout       = params['dropout']
    learning_rate = params['learning_rate']
    num_filters   = [100, 100, 100]
    filter_sizes  = [2,3,4]

    K.set_learning_phase(training)
    
    with tf.device('/cpu:0'):
      embedding = tf.get_variable("word2vec", initializer=word2vec, trainable=False)
      s1 = tf.nn.embedding_lookup(embedding, s1)
      s2 = tf.nn.embedding_lookup(embedding, s2)
      s1 = tf.layers.dropout(s1, 0.2, training=training)
      s2 = tf.layers.dropout(s2, 0.2, training=training)

    # siamese layers
    out1 = cnn_layers(s1, num_filters, filter_sizes)
    out2 = cnn_layers(s1, num_filters, filter_sizes, reuse=True)

    out1 = tf.layers.dropout(out1, 0.5, training=training)
    out2 = tf.layers.dropout(out2, 0.5, training=training)

    merged = concatenate([out1, out2])
    merged = Dense(hidden_size, activation='relu')(merged)
    # merged = Dropout(dropout)(merged)
    # merged = BatchNormalization()(merged)
    
    logits = tf.squeeze(Dense(1)(merged))

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
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
