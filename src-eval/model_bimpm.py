import tensorflow as tf

L = tf.keras.layers
K = tf.keras.backend
TimeDistributed = L.TimeDistributed
Dense = L.Dense
Lambda = L.Lambda
concatenate = L.concatenate
Dropout = L.Dropout
BatchNormalization = L.BatchNormalization

class ModelBiMPM(object):
  def __init__(self, params, word2vec, features, labels, training=False):
      
    len1, len2, s1, s2 = features
    embed_dim     = params['embed_dim']
    hidden_size   = params['hidden_size']
    dropout       = params['dropout']
    learning_rate = params['learning_rate']

    K.set_learning_phase(training)
  
    embedding = tf.get_variable("word2vec", initializer=word2vec, trainable=False)
    with tf.device('/cpu:0'):
      s1 = tf.nn.embedding_lookup(embedding, s1)
      s2 = tf.nn.embedding_lookup(embedding, s2)

    s1 = TimeDistributed(Dense(embed_dim, activation='relu'))(s1)
    s1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(embed_dim, ))(s1)

    s2 = TimeDistributed(Dense(embed_dim, activation='relu'))(s2)
    s2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(embed_dim, ))(s2)

    merged = concatenate([s1,s2])
    merged = Dense(hidden_size, activation='relu')(merged)
    merged = Dropout(dropout)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(hidden_size, activation='relu')(merged)
    merged = Dropout(dropout)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(hidden_size, activation='relu')(merged)
    merged = Dropout(dropout)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(hidden_size, activation='relu')(merged)
    merged = Dropout(dropout)(merged)
    merged = BatchNormalization()(merged)

    logits = tf.squeeze(Dense(1)(merged))

    self.prob = tf.sigmoid(logits)
    self.pred = tf.rint(self.prob)
    self.acc = tf.metrics.accuracy(labels=labels, predictions=self.pred)

    self.loss = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(labels), logits=logits))
      
    if training:
      self.global_step = tf.train.get_or_create_global_step()
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
