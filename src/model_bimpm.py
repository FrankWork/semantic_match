import tensorflow as tf
import numpy as np

L = tf.keras.layers
K = tf.keras.backend
Dense = L.Dense
Dropout = L.Dropout

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

def my_lstm_layer(input_reps, lstm_dim, input_lengths=None, scope_name=None, reuse=False, is_training=True,
                  dropout_rate=0.2, use_cudnn=True):
    '''
    :param inputs: [batch_size, seq_len, feature_dim]
    :param lstm_dim:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :return:
    '''
    use_cudnn = False
    if is_training:
      input_reps = tf.nn.dropout(input_reps, 1-dropout_rate)
    with tf.variable_scope(scope_name, reuse=reuse):
        if use_cudnn:
            inputs = tf.transpose(input_reps, [1, 0, 2])
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, lstm_dim, direction="bidirectional",
                                    name="{}_cudnn_bi_lstm".format(scope_name), dropout=dropout_rate if is_training else 0)
            outputs, _ = lstm(inputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            f_rep = outputs[:, :, 0:lstm_dim]
            b_rep = outputs[:, :, lstm_dim:2*lstm_dim]
        else:
            context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
            context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
            if is_training:
                context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
            context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
            context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

            (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(
                context_lstm_cell_fw, context_lstm_cell_bw, input_reps, dtype=tf.float32,
                sequence_length=input_lengths)  # [batch_size, question_len, context_lstm_dim]
            outputs = tf.concat(axis=2, values=[f_rep, b_rep])
    return (f_rep,b_rep, outputs)

def highway(in_val, output_size, scope='highway', reuse=False):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope, reuse=reuse):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = trans*gate + in_val* (1.0- gate)
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

def cosine_distance(y1,y2, cosine_norm=True, eps=1e-6):
    # cosine_norm = True
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    if not cosine_norm:
        return tf.tanh(cosine_numerator)
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

def multi_perspective_match(feature_dim, s1, s2, cosine_MP_dim=5,
                            scope_name='mp-match', reuse=False):
    '''
        :param s1: [batch_size, len, feature_dim]
        :param s2: [batch_size, len, feature_dim]
        :return:
    '''
    input_shape = tf.shape(s1)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    matching_result = []
    with tf.variable_scope(scope_name, reuse=reuse):
        match_dim = 0

        cosine_value = cosine_distance(s1, s2, cosine_norm=False)
        cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
        matching_result.append(cosine_value)
        match_dim += 1

        w = tf.get_variable("mp_cosine", shape=[cosine_MP_dim, feature_dim], dtype=tf.float32)
        w = tf.expand_dims(w, axis=0)
        w = tf.expand_dims(w, axis=0)
        s1_flat = tf.expand_dims(s1, axis=2)
        s2_flat = tf.expand_dims(s2, axis=2)
        mp_cosine_matching = cosine_distance(tf.multiply(s1_flat, w),
                                                            s2_flat,cosine_norm=False)
        matching_result.append(mp_cosine_matching)
        match_dim += cosine_MP_dim

    matching_result = tf.concat(axis=2, values=matching_result)
    return (matching_result, match_dim)

def calcuate_attention(in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                       att_type='symmetric', att_dim=50, remove_diagnoal=False, 
                       mask1=None, mask2=None, is_training=False, dropout_rate=0.2):
    input_shape = tf.shape(in_value_1)
    batch_size = input_shape[0]
    len_1 = input_shape[1]
    len_2 = tf.shape(in_value_2)[1]

    if is_training:
       in_value_1 = tf.nn.dropout(in_value_1, dropout_rate)
       in_value_2 = tf.nn.dropout(in_value_2, dropout_rate)
    with tf.variable_scope(scope_name):
        # calculate attention ==> a: [batch_size, len_1, len_2]
        atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
        if feature_dim1 == feature_dim2: atten_w2 = atten_w1
        else: atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
        atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]), atten_w1)  # [batch_size*len_1, feature_dim]
        atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
        atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]), atten_w2)  # [batch_size*len_2, feature_dim]
        atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])


        if att_type == 'additive':
            atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
            atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2, name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_2, axis=1, name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
            atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
            atten_value = atten_value + atten_b
            atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1, att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
        else:
            atten_value_1 = tf.tanh(atten_value_1)
            # atten_value_1 = tf.nn.relu(atten_value_1)
            atten_value_2 = tf.tanh(atten_value_2)
            # atten_value_2 = tf.nn.relu(atten_value_2)
            diagnoal_params = tf.get_variable("diagnoal_params", [1, 1, att_dim], dtype=tf.float32)
            atten_value_1 = atten_value_1 * diagnoal_params
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True) # [batch_size, len_1, len_2]

        # normalize
        if remove_diagnoal:
            diagnoal = tf.ones([len_1], tf.float32)  # [len1]
            diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
            diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
            atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
        atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
        if remove_diagnoal: atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))

    return atten_value

def collect_final_step_of_lstm(lstm_representation, lengths):
    # lstm_representation: [batch_size, passsage_length, dim]
    # lengths: [batch_size]
    lengths = tf.cast(lengths, tf.int32)
    lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

    batch_size = tf.shape(lengths)[0]
    batch_nums = tf.range(0, limit=batch_size, dtype=tf.int32) # shape (batch_size)
    indices = tf.stack((batch_nums, lengths), axis=1) # shape (batch_size, 2)
    result = tf.gather_nd(lstm_representation, indices, name='last-forwar-lstm')
    return result # [batch_size, dim]

def match(p, q, p_mask, q_mask, p_len, q_len,
          context_lstm_dim, scope=None, with_full_match=True, 
        is_training=True, dropout_rate=0, forward=True):
    q = q * tf.expand_dims(q_mask, -1) # question
    p = p * tf.expand_dims(p_mask, -1) # passage

    context = [] # all_question_aware_representatins
    dim = 0
    with tf.variable_scope(scope or "match"):
        r = relevancy_matrix(q, p, q_mask, p_mask)

        context.append(tf.reduce_max(r, axis=2, keepdims=True))
        context.append(tf.reduce_mean(r, axis=2, keepdims=True))
        dim += 2
        if with_full_match:
            if forward:
                question_full_rep = collect_final_step_of_lstm(q, q_len - 1)
            else:
                question_full_rep = q[:,0,:]

            passage_len = tf.shape(p)[1]
            question_full_rep = tf.expand_dims(question_full_rep, axis=1)
            question_full_rep = tf.tile(question_full_rep, [1, passage_len, 1])  # [batch_size, pasasge_len, feature_dim]

            (attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                                p, question_full_rep,
                                scope_name='mp-match-full-match')
            context.append(attentive_rep)
            dim += match_dim

        atten_scores = calcuate_attention(p, q, 
                context_lstm_dim, context_lstm_dim,
                scope_name="attention", remove_diagnoal=False, 
                mask1=p_mask, mask2=q_mask, 
                is_training=is_training, dropout_rate=dropout_rate)
        att_question_contexts = tf.matmul(atten_scores, q)
        (attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                p, att_question_contexts,
                scope_name='mp-match-att_question')
        context.append(attentive_rep)
        dim += match_dim

        context = tf.concat(axis=2, values=context)
    return (context, dim)

def bilateral_match(q, p, q_len, p_len, input_dim, is_training, dropout=0.1):
    q_mask = tf.sequence_mask(q_len, dtype=tf.float32)
    p_mask = tf.sequence_mask(p_len, dtype=tf.float32)

    question_aware_representatins = []
    question_aware_dim = 0
    passage_aware_representatins = []
    passage_aware_dim = 0

    # ====word level matching======
    (match_reps, match_dim) = match(p, q, p_mask, q_mask, p_len,
                                q_len, input_dim, scope="word_match_forward",
                                with_full_match=False,  
                                is_training=is_training, dropout_rate=dropout, forward=True)
    question_aware_representatins.append(match_reps)
    question_aware_dim += match_dim

    (match_reps, match_dim) = match(q, p, q_mask, p_mask, q_len,
                                p_len, input_dim, scope="word_match_backward",
                                with_full_match=False,  
                                is_training=is_training, dropout_rate=dropout, forward=False)
    passage_aware_representatins.append(match_reps)
    passage_aware_dim += match_dim

    context_lstm_dim = 100
    aggregation_lstm_dim = 100

    with tf.variable_scope('context_MP_matching'):
        for i in range(1): # support multiple context layer
            with tf.variable_scope('layer-{}'.format(i)):
                # contextual lstm for both passage and question
                q = tf.multiply(q, tf.expand_dims(q_mask, axis=-1))
                p = tf.multiply(p, tf.expand_dims(p_mask, axis=-1))
                (question_context_representation_fw, question_context_representation_bw,
                 q) = my_lstm_layer(
                        q, context_lstm_dim, input_lengths= q_len,scope_name="context_represent",
                        reuse=False, is_training=is_training, dropout_rate=dropout, use_cudnn=True)
                (passage_context_representation_fw, passage_context_representation_bw, 
                 p) = my_lstm_layer(
                        p, context_lstm_dim, input_lengths=p_len, scope_name="context_represent",
                        reuse=True, is_training=is_training, dropout_rate=dropout, use_cudnn=True)

                # Multi-perspective matching
                with tf.variable_scope('left_MP_matching'):
                    (match_reps, match_dim) = match(passage_context_representation_fw,
                                question_context_representation_fw, p_mask, q_mask, p_len,
                                q_len, context_lstm_dim, scope="forward_match",
                                with_full_match=True,  
                                is_training=is_training, dropout_rate=dropout, forward=True)
                    question_aware_representatins.append(match_reps)
                    question_aware_dim += match_dim
                    (match_reps, match_dim) = match(passage_context_representation_bw,
                                question_context_representation_bw, p_mask, q_mask, p_len,
                                q_len, context_lstm_dim, scope="backward_match",
                                with_full_match=True,  
                                is_training=is_training, dropout_rate=dropout, forward=False)
                    question_aware_representatins.append(match_reps)
                    question_aware_dim += match_dim

                with tf.variable_scope('right_MP_matching'):
                    (match_reps, match_dim) = match(question_context_representation_fw,
                                passage_context_representation_fw, q_mask, p_mask, q_len,
                                p_len, context_lstm_dim, scope="forward_match",
                                with_full_match=True,  
                                is_training=is_training, dropout_rate=dropout, forward=True)
                    passage_aware_representatins.append(match_reps)
                    passage_aware_dim += match_dim
                    (match_reps, match_dim) = match(question_context_representation_bw,
                                passage_context_representation_bw, q_mask, p_mask, q_len,
                                p_len, context_lstm_dim, scope="backward_match",
                                with_full_match=True,  
                                is_training=is_training, dropout_rate=dropout, forward=False)
                    passage_aware_representatins.append(match_reps)
                    passage_aware_dim += match_dim

    question_aware_representatins = tf.concat(axis=2, values=question_aware_representatins) # [batch_size, passage_len, question_aware_dim]
    passage_aware_representatins = tf.concat(axis=2, values=passage_aware_representatins) # [batch_size, question_len, question_aware_dim]

    if is_training:
        question_aware_representatins = tf.nn.dropout(question_aware_representatins, (1 - dropout))
        passage_aware_representatins = tf.nn.dropout(passage_aware_representatins, (1 - dropout))
        
    # ======Highway layer======
    with tf.variable_scope("left_matching_highway"):
        question_aware_representatins = highway(question_aware_representatins, question_aware_dim)
    with tf.variable_scope("right_matching_highway"):
        passage_aware_representatins = highway(passage_aware_representatins, passage_aware_dim)

    #========Aggregation Layer======
    aggregation_representation = []
    aggregation_dim = 0
    
    qa_aggregation_input = question_aware_representatins
    pa_aggregation_input = passage_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        for i in range(1): # support multiple aggregation layer
            qa_aggregation_input = tf.multiply(qa_aggregation_input, tf.expand_dims(p_mask, axis=-1))
            (fw_rep, bw_rep, cur_aggregation_representation) = my_lstm_layer(
                        qa_aggregation_input, aggregation_lstm_dim, input_lengths=p_len, scope_name='left_layer-{}'.format(i),
                        reuse=False, is_training=is_training, dropout_rate=dropout,use_cudnn=True)
            fw_rep = collect_final_step_of_lstm(fw_rep, p_len - 1)
            bw_rep = bw_rep[:, 0, :]
            aggregation_representation.append(fw_rep)
            aggregation_representation.append(bw_rep)
            aggregation_dim += 2* aggregation_lstm_dim
            qa_aggregation_input = cur_aggregation_representation# [batch_size, passage_len, 2*aggregation_lstm_dim]

            pa_aggregation_input = tf.multiply(pa_aggregation_input, tf.expand_dims(q_mask, axis=-1))
            (fw_rep, bw_rep, cur_aggregation_representation) = my_lstm_layer(
                        pa_aggregation_input, aggregation_lstm_dim,
                        input_lengths=q_len, scope_name='right_layer-{}'.format(i),
                        reuse=False, is_training=is_training, dropout_rate=dropout, use_cudnn=True)
            fw_rep = collect_final_step_of_lstm(fw_rep, q_len - 1)
            bw_rep = bw_rep[:, 0, :]
            aggregation_representation.append(fw_rep)
            aggregation_representation.append(bw_rep)
            aggregation_dim += 2* aggregation_lstm_dim
            pa_aggregation_input = cur_aggregation_representation# [batch_size, passage_len, 2*aggregation_lstm_dim]

    aggregation_representation = tf.concat(axis=1, values=aggregation_representation) # [batch_size, aggregation_dim]

    # ======Highway layer======
    with tf.variable_scope("aggregation_highway"):
        agg_shape = tf.shape(aggregation_representation)
        batch_size = agg_shape[0]
        aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
        aggregation_representation = highway(aggregation_representation, aggregation_dim)
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
    s2 = highway(s2, embed_dim, reuse=True)

    matched, dim = bilateral_match(s1, s2, len1, len2, embed_dim, training)

    matched  = Dense(dim/2, activation='relu')(matched)
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
