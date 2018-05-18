import tensorflow as tf
import numpy as np

L = tf.keras.layers
K = tf.keras.backend
Dense = L.Dense
Dropout = L.Dropout
eps = 1e-6

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


def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = trans*gate + in_val* (1.0- gate)
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

def highway(in_val, output_size, num_layers=1, scope='highway'):
  for i in range(num_layers):
    cur_scope = scope + "-{}".format(i)
    in_val = highway_layer(in_val, output_size, scope=cur_scope)
  return in_val

def cosine_distance(y1,y2):
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps)) 
    return cosine_numerator / y1_norm / y2_norm

def relevancy_matrix(y1, y2, y1_mask, y2_mask):
    y1_tmp = tf.expand_dims(y1, 1) # [batch_size, 1, question_len, dim]
    y2_tmp = tf.expand_dims(y2, 2) # [batch_size, passage_len, 1, dim]
    r = cosine_distance(y1_tmp,y2_tmp) # [batch_size, passage_len, question_len]

    r = tf.multiply(r, tf.expand_dims(y1_mask, 1))
    r = tf.multiply(r, tf.expand_dims(y2_mask, 2))
    return r


def match(p, q, p_len, q_len, p_mask, q_mask,
          context_lstm_dim, scope=None,
        with_full_match=True, 
        with_attentive_match=True, 
        is_training=True, options=None, dropout_rate=0, forward=True):
    q = q * tf.expand_dims(q_mask, -1) # question
    p = p * tf.expand_dims(p_mask, -1) # passage

    context = [] # all_question_aware_representatins
    dim = 0
    with tf.variable_scope(scope or "match"):
        r = relevancy_matrix(q, p, q_mask, p_mask)

        context.append(tf.reduce_max(r, axis=2,keep_dims=True))
        context.append(tf.reduce_mean(r, axis=2,keep_dims=True))
        dim += 2
        if with_full_match:
            if forward:
                question_full_rep = layer_utils.collect_final_step_of_lstm(question_reps, question_lengths - 1)
            else:
                question_full_rep = question_reps[:,0,:]

            passage_len = tf.shape(passage_reps)[1]
            question_full_rep = tf.expand_dims(question_full_rep, axis=1)
            question_full_rep = tf.tile(question_full_rep, [1, passage_len, 1])  # [batch_size, pasasge_len, feature_dim]

            (attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                                passage_reps, question_full_rep, is_training=is_training, dropout_rate=options.dropout_rate,
                                options=options, scope_name='mp-match-full-match')
            context.append(attentive_rep)
            dim += match_dim

        if with_attentive_match:
            atten_scores = layer_utils.calcuate_attention(passage_reps, question_reps, context_lstm_dim, context_lstm_dim,
                    scope_name="attention", att_type=options.att_type, att_dim=options.att_dim,
                    remove_diagnoal=False, mask1=passage_mask, mask2=question_mask, is_training=is_training, dropout_rate=dropout_rate)
            att_question_contexts = tf.matmul(atten_scores, question_reps)
            (attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                    passage_reps, att_question_contexts, is_training=is_training, dropout_rate=options.dropout_rate,
                    options=options, scope_name='mp-match-att_question')
            context.append(attentive_rep)
            dim += match_dim

        context = tf.concat(axis=2, values=context)
    return (context, dim)


def bilateral_match(in_question_repres, in_passage_repres,
                        question_lengths, passage_lengths, 
                        question_mask, passage_mask, 
                        input_dim, is_training, options=None):

    question_aware_representatins = []
    question_aware_dim = 0
    passage_aware_representatins = []
    passage_aware_dim = 0

    # ====word level matching======
    (match_reps, match_dim) = match(in_passage_repres, in_question_repres, passage_mask, question_mask, passage_lengths,
                                question_lengths, input_dim, scope="word_match_forward",
                                with_full_match=False,  
                                with_attentive_match=options.with_attentive_match,
                                  
                                is_training=is_training, options=options, dropout_rate=options.dropout_rate, forward=True)
    question_aware_representatins.append(match_reps)
    question_aware_dim += match_dim

    (match_reps, match_dim) = match(in_question_repres, in_passage_repres, question_mask, passage_mask, question_lengths,
                                passage_lengths, input_dim, scope="word_match_backward",
                                with_full_match=False,  
                                with_attentive_match=options.with_attentive_match,
                                  
                                is_training=is_training, options=options, dropout_rate=options.dropout_rate, forward=False)
    passage_aware_representatins.append(match_reps)
    passage_aware_dim += match_dim

    with tf.variable_scope('context_MP_matching'):
        for i in xrange(options.context_layer_num): # support multiple context layer
            with tf.variable_scope('layer-{}'.format(i)):
                # contextual lstm for both passage and question
                in_question_repres = tf.multiply(in_question_repres, tf.expand_dims(question_mask, axis=-1))
                in_passage_repres = tf.multiply(in_passage_repres, tf.expand_dims(passage_mask, axis=-1))
                (question_context_representation_fw, question_context_representation_bw,
                 in_question_repres) = layer_utils.my_lstm_layer(
                        in_question_repres, options.context_lstm_dim, input_lengths= question_lengths,scope_name="context_represent",
                        reuse=False, is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
                (passage_context_representation_fw, passage_context_representation_bw, 
                 in_passage_repres) = layer_utils.my_lstm_layer(
                        in_passage_repres, options.context_lstm_dim, input_lengths=passage_lengths, scope_name="context_represent",
                        reuse=True, is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)

                # Multi-perspective matching
                with tf.variable_scope('left_MP_matching'):
                    (match_reps, match_dim) = match(passage_context_representation_fw,
                                question_context_representation_fw, passage_mask, question_mask, passage_lengths,
                                question_lengths, options.context_lstm_dim, scope="forward_match",
                                with_full_match=True,  
                                with_attentive_match=options.with_attentive_match,
                                  
                                is_training=is_training, options=options, dropout_rate=options.dropout_rate, forward=True)
                    question_aware_representatins.append(match_reps)
                    question_aware_dim += match_dim
                    (match_reps, match_dim) = match(passage_context_representation_bw,
                                question_context_representation_bw, passage_mask, question_mask, passage_lengths,
                                question_lengths, options.context_lstm_dim, scope="backward_match",
                                with_full_match=True,  
                                with_attentive_match=options.with_attentive_match,
                                  
                                is_training=is_training, options=options, dropout_rate=options.dropout_rate, forward=False)
                    question_aware_representatins.append(match_reps)
                    question_aware_dim += match_dim

                with tf.variable_scope('right_MP_matching'):
                    (match_reps, match_dim) = match(question_context_representation_fw,
                                passage_context_representation_fw, question_mask, passage_mask, question_lengths,
                                passage_lengths, options.context_lstm_dim, scope="forward_match",
                                with_full_match=True,  
                                with_attentive_match=options.with_attentive_match,
                                  
                                is_training=is_training, options=options, dropout_rate=options.dropout_rate, forward=True)
                    passage_aware_representatins.append(match_reps)
                    passage_aware_dim += match_dim
                    (match_reps, match_dim) = match(question_context_representation_bw,
                                passage_context_representation_bw, question_mask, passage_mask, question_lengths,
                                passage_lengths, options.context_lstm_dim, scope="backward_match",
                                with_full_match=True,  
                                with_attentive_match=options.with_attentive_match,
                                  
                                is_training=is_training, options=options, dropout_rate=options.dropout_rate, forward=False)
                    passage_aware_representatins.append(match_reps)
                    passage_aware_dim += match_dim

    question_aware_representatins = tf.concat(axis=2, values=question_aware_representatins) # [batch_size, passage_len, question_aware_dim]
    passage_aware_representatins = tf.concat(axis=2, values=passage_aware_representatins) # [batch_size, question_len, question_aware_dim]

    if is_training:
        question_aware_representatins = tf.nn.dropout(question_aware_representatins, (1 - options.dropout_rate))
        passage_aware_representatins = tf.nn.dropout(passage_aware_representatins, (1 - options.dropout_rate))
        
    # ======Highway layer======
    if options.with_match_highway:
        with tf.variable_scope("left_matching_highway"):
            question_aware_representatins = multi_highway_layer(question_aware_representatins, question_aware_dim,
                                                                options.highway_layer_num)
        with tf.variable_scope("right_matching_highway"):
            passage_aware_representatins = multi_highway_layer(passage_aware_representatins, passage_aware_dim,
                                                               options.highway_layer_num)

    #========Aggregation Layer======
    aggregation_representation = []
    aggregation_dim = 0
    
    qa_aggregation_input = question_aware_representatins
    pa_aggregation_input = passage_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        for i in xrange(options.aggregation_layer_num): # support multiple aggregation layer
            qa_aggregation_input = tf.multiply(qa_aggregation_input, tf.expand_dims(passage_mask, axis=-1))
            (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
                        qa_aggregation_input, options.aggregation_lstm_dim, input_lengths=passage_lengths, scope_name='left_layer-{}'.format(i),
                        reuse=False, is_training=is_training, dropout_rate=options.dropout_rate,use_cudnn=options.use_cudnn)
            fw_rep = layer_utils.collect_final_step_of_lstm(fw_rep, passage_lengths - 1)
            bw_rep = bw_rep[:, 0, :]
            aggregation_representation.append(fw_rep)
            aggregation_representation.append(bw_rep)
            aggregation_dim += 2* options.aggregation_lstm_dim
            qa_aggregation_input = cur_aggregation_representation# [batch_size, passage_len, 2*aggregation_lstm_dim]

            pa_aggregation_input = tf.multiply(pa_aggregation_input, tf.expand_dims(question_mask, axis=-1))
            (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
                        pa_aggregation_input, options.aggregation_lstm_dim,
                        input_lengths=question_lengths, scope_name='right_layer-{}'.format(i),
                        reuse=False, is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
            fw_rep = layer_utils.collect_final_step_of_lstm(fw_rep, question_lengths - 1)
            bw_rep = bw_rep[:, 0, :]
            aggregation_representation.append(fw_rep)
            aggregation_representation.append(bw_rep)
            aggregation_dim += 2* options.aggregation_lstm_dim
            pa_aggregation_input = cur_aggregation_representation# [batch_size, passage_len, 2*aggregation_lstm_dim]

    aggregation_representation = tf.concat(axis=1, values=aggregation_representation) # [batch_size, aggregation_dim]

    # ======Highway layer======
    if options.with_aggregation_highway:
        with tf.variable_scope("aggregation_highway"):
            agg_shape = tf.shape(aggregation_representation)
            batch_size = agg_shape[0]
            aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
            aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, options.highway_layer_num)
            aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])
    
    return (aggregation_representation, aggregation_dim)



class ModelBiMPM(object):
  def __init__(self, params, word2vec, features, labels, training=False):
    len1, len2, s1, s2 = features
    embed_dim     = params['embed_dim']
    hidden_size   = params['hidden_size']
    dropout       = params['dropout']
    input_keep    = 0.8
    learning_rate = 0.0005
    max_norm      = 10

    K.set_learning_phase(training)

    with tf.device('/cpu:0'):
      embedding = tf.get_variable("word2vec", initializer=word2vec, trainable=False)
      s1 = tf.nn.embedding_lookup(embedding, s1)
      s2 = tf.nn.embedding_lookup(embedding, s2)
    if training:
      s1 = tf.nn.dropout(s1, input_keep)
      s2 = tf.nn.dropout(s2, input_keep)

    s1 = highway(s1, embed_dim)
    s2 = highway(s2, embed_dim)

    matched = bilateral_match(s1, s2, len1, len2, training)

    matched  = Dense(hidden_size/2, activation='relu')(matched)
    matched  = Dropout(dropout)(matched)

    logits = tf.squeeze(Dense(1)(matched))

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
