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
    
    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                  bw_cell,
                                                  inputs,
                                                  sequence_length=length,
                                                  dtype=tf.float32)
    return tf.concat(bi_outputs, -1), bi_state

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

def avg_checkpoints(model_dir, num_last_checkpoints, global_step,
                    global_step_name):
  """Average the last N checkpoints in the model_dir."""
  checkpoint_state = tf.train.get_checkpoint_state(model_dir)
  if not checkpoint_state:
    utils.print_out("# No checkpoint file found in directory: %s" % model_dir)
    return None

  # Checkpoints are ordered from oldest to newest.
  checkpoints = (
      checkpoint_state.all_model_checkpoint_paths[-num_last_checkpoints:])

  if len(checkpoints) < num_last_checkpoints:
    utils.print_out(
        "# Skipping averaging checkpoints because not enough checkpoints is "
        "avaliable."
    )
    return None

  avg_model_dir = os.path.join(model_dir, "avg_checkpoints")
  if not tf.gfile.Exists(avg_model_dir):
    utils.print_out(
        "# Creating new directory %s for saving averaged checkpoints." %
        avg_model_dir)
    tf.gfile.MakeDirs(avg_model_dir)

  utils.print_out("# Reading and averaging variables in checkpoints:")
  var_list = tf.contrib.framework.list_variables(checkpoints[0])
  var_values, var_dtypes = {}, {}
  for (name, shape) in var_list:
    if name != global_step_name:
      var_values[name] = np.zeros(shape)

  for checkpoint in checkpoints:
    utils.print_out("    %s" % checkpoint)
    reader = tf.contrib.framework.load_checkpoint(checkpoint)
    for name in var_values:
      tensor = reader.get_tensor(name)
      var_dtypes[name] = tensor.dtype
      var_values[name] += tensor

  for name in var_values:
    var_values[name] /= len(checkpoints)

  # Build a graph with same variables in the checkpoints, and save the averaged
  # variables into the avg_model_dir.
  with tf.Graph().as_default():
    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[name])
        for v in var_values
    ]

    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    global_step_var = tf.Variable(
        global_step, name=global_step_name, trainable=False)
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                             six.iteritems(var_values)):
        sess.run(assign_op, {p: value})

      # Use the built saver to save the averaged checkpoint. Only keep 1
      # checkpoint and the best checkpoint will be moved to avg_best_metric_dir.
      saver.save(
          sess,
          os.path.join(avg_model_dir, "translate.ckpt"))

  return avg_model_dir

def decode(encoder_outputs, encoder_state, src_len, targets, tgt_len, 
          hidden_size, vocab_size, training, dropout_rate=0.1, 
          name="decoder", reuse=False):
  with tf.variable_scope(name, reuse=reuse) as decoder_scope:
    LSTM = tf.nn.rnn_cell.BasicLSTMCell
    decode_cell = LSTM(hidden_size, name='de')
    if training:
      decode_cell = tf.nn.rnn_cell.DropoutWrapper(decode_cell, 
                                        output_keep_prob=(1 - dropout_rate))
    
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                              hidden_size,
                              encoder_outputs,
                              memory_sequence_length=src_len,
                              normalize=True)

    decode_cell = tf.contrib.seq2seq.AttentionWrapper(
                              decode_cell,
                              attention_mechanism,
                              attention_layer_size=hidden_size,
                              alignment_history=False,
                              output_attention=True,
                              name="attention")

    # Helper
    helper = tf.contrib.seq2seq.TrainingHelper(targets, tgt_len,)
    # Decoder
    my_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, helper, encoder_state,)

    # Dynamic decoding
    outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                            my_decoder, swap_memory=True, scope=decoder_scope)

    sample_id = outputs.sample_id

    logits = tf.layers.dense(outputs.rnn_output, vocab_size,
                              use_bias=False, name="output_projection")

  return logits, sample_id, final_context_state

def seq_loss(logits, targets, length):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=targets, logits=logits)
    target_weights = tf.sequence_mask(length, dtype=logits.dtype)
    batch_size = tf.size(length)

    loss = tf.reduce_sum(
                        crossent * target_weights) / tf.to_float(batch_size)
    return loss

def seq2seq(src, src_len, tgt, tgt_len, labels, hidden_size, 
                          training, dropout=0.1, reuse=False, vocab_size=5924):
  src_out, src_state = biRNN(src, src_len, hidden_size, training, dropout, 
                             "encode_seq", reuse=reuse)
  logits1, _, _ = decode(src_out, src_state, src_len, tgt, tgt_len, 
                          hidden_size, vocab_size, training, 0.1, 
                          "decode_seq", reuse)
  loss1 = seq_loss(logits1, labels, tgt_len)
  return src_out, src_state, loss1

class ModelSeq(object):
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
      q1 = tf.nn.embedding_lookup(embedding, s1)
      q2 = tf.nn.embedding_lookup(embedding, s2)
    if training:
      q1 = tf.nn.dropout(q1, input_keep)
      q2 = tf.nn.dropout(q2, input_keep)

    q1 = highway(q1, embed_dim, 'highway_in', training)
    q2 = highway(q2, embed_dim, 'highway_in', training, reuse=True)

    #=======================
    # Seq Model
    #=======================
    out1, state1, loss1 = seq2seq(q1, len1, q2, len2, s2, 
                                        hidden_size, training, 0.1, reuse=False)
    out2, state2, loss2 = seq2seq(q2, len2, q1, len1, s1, 
                                        hidden_size, training, 0.1, reuse=True)

    #=======================
    # ESIM Model
    #=======================
    # Encoding
    q1_encoded, _ = biRNN(s1, len1, hidden_size, training, 0.1, "encode")
    q2_encoded, _ = biRNN(s2, len2, hidden_size, training, 0.1, "encode", reuse=True)
    
    # Alignment
    q1_aligned, q2_aligned = align(q1_encoded, q2_encoded)
    
    # Compare
    q1_combined = concatenate([q1_encoded, q2_aligned, 
                        tf.abs(q1_encoded-q2_aligned), q1_encoded*q2_aligned],
                        out1)
    q2_combined = concatenate([q2_encoded, q1_aligned, 
                        tf.abs(q2_encoded-q1_aligned), q2_encoded*q1_aligned],
                        out2)

    q1_proj = proj(q1_combined, hidden_size, 0.5)
    q2_proj = proj(q2_combined, hidden_size, 0.5)

    q1_compare, _ = biRNN(q1_proj, len1, hidden_size, training, 0.1, "compare")
    q2_compare, _ = biRNN(q2_proj, len2, hidden_size, training, 0.1, "compare", reuse=True)
    
    # Aggregate
    x = aggregate(q1_compare, q2_compare, hidden_size, training)
        
    logits = tf.squeeze(Dense(1)(x))

    #=======================
    # prediction and loss
    #=======================

    self.prob = tf.sigmoid(logits)
    self.pred = tf.rint(self.prob)
    self.acc = tf.metrics.accuracy(labels=labels, predictions=self.pred)

    self.loss = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=tf.to_float(labels), logits=logits))
    l2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                    if 'bias' not in v.name ]) * l2_coef
    self.loss += l2
    
    #=======================
    # optimizer
    #=======================
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

