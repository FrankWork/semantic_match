import tensorflow as tf
from keras.layers import *


INF = 1e30


def biRNN(inputs, length, hidden_size, training, dropout_rate=0.1, 
          name="biRNN", num_layers=1, reuse=False): 
  keep_prob = 1 - dropout_rate

  cell = tf.nn.rnn_cell.BasicLSTMCell
  DropoutWrapper = tf.nn.rnn_cell.DropoutWrapper
  MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell

  with tf.variable_scope(name, reuse=reuse):
    fw_cells = []
    bw_cells = []
    for i in range(num_layers):
      fw_cell = cell(hidden_size, name='fw')
      bw_cell = cell(hidden_size, name='bw')
      if training:
        fw_cell = DropoutWrapper(fw_cell, keep_prob)
        bw_cell = DropoutWrapper(bw_cell, keep_prob)
      fw_cells.append(fw_cell)
      bw_cells.append(bw_cell)
    
    fw_cells = MultiRNNCell([fw_cells])
    bw_cells = MultiRNNCell([bw_cells])
    
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                bw_cell,
                                                inputs,
                                                sequence_length=length,
                                                dtype=tf.float32)
    output = tf.concat([outputs[0], outputs[1]], axis=2)
  return output
  
def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        if is_train:
          args = tf.nn.dropout(args, keep_prob, noise_shape=noise_shape)
    return args


def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val

def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, 
                  scope="dot_attention", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
      d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
      d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
      JX = tf.shape(inputs)[1]
      dense = tf.layers.dense
      relu = tf.nn.relu

      with tf.variable_scope("attention"):
          inputs_ = dense(d_inputs, hidden, relu, use_bias=False, name="inputs")
          memory_ = dense(d_memory, hidden, relu, use_bias=False, name="memory")
          outputs = tf.matmul(inputs_, tf.transpose(
              memory_, [0, 2, 1])) / (hidden ** 0.5)
          mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
          logits = tf.nn.softmax(softmax_mask(outputs, mask))
          outputs = tf.matmul(logits, memory)
          res = tf.concat([inputs, outputs, inputs-outputs, inputs*outputs], axis=2)

      with tf.variable_scope("gate"):
          dim = res.get_shape().as_list()[-1]
          d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
          gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
          return res * gate

def aggregate(input_1, input_2, num_dense=200, dropout_rate=0.5):
    feat1 = concatenate([GlobalAvgPool1D()(input_1), GlobalMaxPool1D()(input_1)])
    feat2 = concatenate([GlobalAvgPool1D()(input_2), GlobalMaxPool1D()(input_2)])

    x = concatenate([feat1, feat2])
    x = Dropout(dropout_rate)(x)
    
    x = Dense(num_dense, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    return x   

class ModelRNet(object):
  def __init__(self, params, word2vec, features, labels, training=False):
    len1, len2, s1, s2 = features
    embed_dim     = params['embed_dim']
    hidden_size   = params['hidden_size']
    dropout       = params['dropout']
    input_keep    = 0.8
    learning_rate = 0.001
    max_norm      = 10
    l2_coef       = 0.0001
    nN = 30797 
    nP = 8549

    K.set_learning_phase(training)

    with tf.device('/cpu:0'):
      embedding = tf.get_variable("word2vec", initializer=word2vec, trainable=False)
      s1 = tf.nn.embedding_lookup(embedding, s1)
      s2 = tf.nn.embedding_lookup(embedding, s2)
    if training:
      s1 = tf.nn.dropout(s1, input_keep)
      s2 = tf.nn.dropout(s2, input_keep)

    # Encoding
    c_encoded = biRNN(s1, len1, hidden_size, training, 0.2, "encode")
    q_encoded = biRNN(s2, len2, hidden_size, training, 0.2, "encode", reuse=True)
    
    c_mask = tf.sequence_mask(len1, dtype=tf.float32)
    q_mask = tf.sequence_mask(len2, dtype=tf.float32)

    # att
    c_att = dot_attention(c_encoded, q_encoded, q_mask, hidden_size, 0.8, training, "att")
    c_match = biRNN(c_att, len1, hidden_size, training, 0.2, "rnn_att")
    
    q_att = dot_attention(q_encoded, c_encoded, c_mask, hidden_size, 0.8, training, "att", reuse=True)
    q_match = biRNN(q_att, len2, hidden_size, training, 0.2, "rnn_att", reuse=True)

    # # match
    # qc_self = dot_attention(qc_att, qc_att, c_mask, hidden_size, 0.8, training, "self")
    # qc_match = biRNN(qc_self, len1, hidden_size, training, 0.2, "match")

    # cq_self = dot_attention(cq_att, cq_att, q_mask, hidden_size, 0.8, training, "self", reuse=True)
    # cq_match = biRNN(cq_self, len2, hidden_size, training, 0.2, "match", reuse=True)

    # Aggregate
    with tf.name_scope('l2_norm'):
      x = aggregate(c_match, q_match)

      logits = tf.squeeze(Dense(1)(x))

    self.prob = tf.sigmoid(logits)
    self.pred = tf.rint(self.prob)
    self.acc = tf.metrics.accuracy(labels=labels, predictions=self.pred)

    self.loss = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=tf.to_float(labels), logits=logits))
    l2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables("l2_norm")
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
    