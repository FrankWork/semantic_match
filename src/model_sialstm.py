import tensorflow as tf

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

def rnn_layer(inputs, length, hidden_size, 
              dropout, training,
              bidirectional=True, cell_type='GRU', reuse=False):
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

class ModelSiameseLSTM(object):
  def __init__(self, params, word2vec, features, labels, training=False):
    len1, len2, s1, s2 = features
    embed_dim     = params['embed_dim']
    hidden_size   = params['hidden_size']
    dropout       = params['dropout']
    learning_rate = params['learning_rate']
    max_norm      = params['max_norm']
    rnn_dropout   = params['rnn_dropout']
    
    cell_type     = "LSTM"

    embedding = tf.get_variable("word2vec", initializer=word2vec, trainable=False)
    with tf.device('/cpu:0'):
      s1 = tf.nn.embedding_lookup(embedding, s1)
      s2 = tf.nn.embedding_lookup(embedding, s2)

    # siamese layers
    outputs_sen1 = rnn_layer(s1, len1, hidden_size, rnn_dropout, training, cell_type)
    outputs_sen2 = rnn_layer(s2, len2, hidden_size, rnn_dropout, training, cell_type, reuse=True)

    out1 = tf.reduce_max(outputs_sen1, axis=1)
    out2 = tf.reduce_max(outputs_sen2, axis=1)

    self.prob = tf.squeeze(manhattan_similarity(out1, out2))
    self.loss = tf.losses.mean_squared_error(labels, self.prob)

    self.pred = tf.rint(self.prob)
    self.acc = tf.metrics.accuracy(labels=labels, predictions=self.pred)

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