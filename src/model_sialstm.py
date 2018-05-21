import tensorflow as tf

L = tf.keras.layers
K = tf.keras.backend
Dense = L.Dense
Dropout = L.Dropout
BatchNormalization = L.BatchNormalization

def batch_norm(x, is_training):
  return tf.contrib.layers.batch_norm(inputs=x, 
                            updates_collections=None, 
                            is_training=is_training)

def manhattan_distance(x1, x2):
  """
  Also known as l1 norm.
  Equation: sum(|x1 - x2|)
  Example:
      x1 = [1,2,3]
      x2 = [3,2,1]
      MD = (|1 - 3|) + (|2 - 2|) + (|3 - 1|) = 4
  Args:
      x1: x1 input vector
      x2: x2 input vector

  Returns: Manhattan distance between x1 and x2. Value grater than or equal to 0.

  """
  return tf.reduce_sum(tf.abs(x1 - x2), axis=1, keep_dims=True)

def manhattan_similarity(x1, x2):
  """
  Similarity function based on manhattan distance and exponential function.
  Args:
      x1: x1 input vector
      x2: x2 input vector

  Returns: Similarity measure in range between 0...1,
  where 1 means full similarity and 0 means no similarity at all.

  """
  with tf.name_scope('manhattan_similarity'):
    manhattan_sim = tf.exp(-manhattan_distance(x1, x2))
  return manhattan_sim

def rnn_layer(inputs, length, hidden_size, dropout, training,
              bidirectional=True, cell_type='LSTM', reuse=False):
  rnn_cells = {
    'GRU': tf.nn.rnn_cell.GRUCell,
    'LSTM': tf.nn.rnn_cell.BasicLSTMCell,
  }
  keep_prob = 1.0 - dropout
  with tf.variable_scope('recurrent', reuse=reuse):
    cell = rnn_cells[cell_type]
    fw_rnn_cell = cell(hidden_size)
    
    if training:
      fw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(fw_rnn_cell,
                                        output_keep_prob=keep_prob)
    if bidirectional:
      bw_rnn_cell = cell(hidden_size)
      if training:
        bw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(bw_rnn_cell,
                                        output_keep_prob=keep_prob)
      rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell,
                                                        bw_rnn_cell,
                                                        inputs,
                                                        sequence_length=length,
                                                        dtype=tf.float32)
      output = tf.concat([rnn_outputs[0], rnn_outputs[1]], axis=2)
    else:
      output, _ = tf.nn.dynamic_rnn(fw_rnn_cell,
                                    inputs,
                                    sequence_length=length,
                                    dtype=tf.float32)
  return output

def baisc_attention(x):
  alpha = tf.layers.dense(tf.nn.tanh(x), 1, use_bias=False) # b,n,1
  alpha = tf.nn.softmax(alpha, axis=1)
  vec = tf.matmul(alpha, x, transpose_a=True)
  vec = tf.squeeze(vec, axis=[1])
  return vec

class ModelSiameseLSTM(object):
  def __init__(self, params, word2vec, features, labels, training=False):
    len1, len2, s1, s2 = features
    embed_dim     = params['embed_dim']
    hidden_size   = params['hidden_size']
    dropout       = params['dropout']
    rnn_dropout   = 0.1
    learning_rate = 0.0005
    max_norm      = 10
    l2_coef       = 0.0001
    K.set_learning_phase(training)

    embedding = tf.get_variable("word2vec", initializer=word2vec, trainable=True)
    with tf.device('/cpu:0'):
      s1 = tf.nn.embedding_lookup(embedding, s1)
      s2 = tf.nn.embedding_lookup(embedding, s2)

    # siamese layers
    out1 = rnn_layer(s1, len1, hidden_size, rnn_dropout, training)
    out2 = rnn_layer(s2, len2, hidden_size, rnn_dropout, training, reuse=True)

    out1 = tf.reduce_max(out1, axis=1)
    out2 = tf.reduce_max(out2, axis=1)

    # self.prob = tf.squeeze(manhattan_similarity(out1, out2))
    # self.loss = tf.losses.mean_squared_error(labels, self.prob)

    matched  = tf.concat([out1, out2], axis=1)
    # matched  = baisc_attention(matched)
    matched  = Dense(hidden_size*2, activation='relu')(matched)
    matched  = Dropout(dropout)(matched)

    # matched = Dense(hidden_size*2, activation='relu')(matched)
    # matched = Dropout(dropout)(matched)
    # matched = batch_norm(matched, training)

    # matched = Dense(hidden_size, activation='relu')(matched)
    # matched = Dropout(dropout)(matched)
    # matched = batch_norm(matched, training)

    # matched = Dense(hidden_size/2, activation='relu')(matched)
    # matched = Dropout(dropout)(matched)
    # matched = batch_norm(matched, training)

    # matched = Dense(hidden_size/2, activation='relu')(matched)
    # matched = Dropout(dropout)(matched)
    # matched = batch_norm(matched, training)

    logits = tf.squeeze(Dense(1)(matched))

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
        self.gradients = gradients
        gradients, _ = tf.clip_by_global_norm(gradients, max_norm)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables), 
                                                  global_step=self.global_step)
        # self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
