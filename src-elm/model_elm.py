import tensorflow as tf
import numpy as np
import os
import random
from tqdm import trange
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

embed_dim     = 300
hidden_size   = 1000
input_keep    = 0.8
learning_rate = 1e-3
max_norm      = 10
l2_coef       = 1e-5
n_vocab       = 4000
n_gpu         = 4

def shape_list(x):
  """
  deal with dynamic shape in tensorflow cleanly
  """
  ps = x.get_shape().as_list()
  ts = tf.shape(x)
  return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def language_model(in_tensors, training=False, reuse=None):
  X, L = in_tensors
  K.set_learning_phase(training)

  M = tf.to_float(tf.sequence_mask(L))

  with tf.variable_scope('shared', reuse=reuse):
    with tf.device('/cpu:0'):
      embedding = tf.get_variable("word2vec", [n_vocab, embed_dim])
      x = tf.nn.embedding_lookup(embedding, X)
    if training:
      x = tf.nn.dropout(x, input_keep)
    x = highway(x, embed_dim, 'highway_in', training)

    # Encoding
    h = biRNN(x, L, embed_dim, training, 0.1, "encode")

    lm_h = tf.reshape(h[:, :-1], [-1, embed_dim]) # remove the last token of lm_h
    lm_logits = tf.matmul(lm_h, embedding, transpose_b=True)
    lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=lm_logits, 
                                    labels=tf.reshape(X[:, 1:, 0], [-1]))
    lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
    lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)
  
    return tf.reduce_mean(lm_losses)

def average_grads(tower_grads):
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad / len(grad_and_vars)

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(indices, 0)
        values = tf.concat(values, 0)
        return tf.IndexedSlices(values, indices, grad_and_vars[0][0].dense_shape)

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:
            grad = None
        elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
            grad = average_sparse(grad_and_vars)
        else:
            grad = average_dense(grad_and_vars)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def assign_to_gpu(gpu=0, ps_dev="/device:CPU:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return ps_dev
        else:
            return "/gpu:%d" % gpu
    return _assign

def mgpu_train_op_lm(in_tensors):
    gpu_ops = []
    gpu_grads = []

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.minimum(0.0005, 
        0.001 / tf.log(999.) * tf.log(tf.cast(global_step, tf.float32) + 1))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # divide evenly
    split_ts = (tf.split(x, n_gpu, 0) for x in in_tensors)
    for i, xs in enumerate(zip(*split_ts)): # n_gpu times
        do_reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
            lm_loss = language_model(xs, training=True, reuse=do_reuse)
            
            grads, params = zip(*optimizer.compute_gradients(lm_loss))
            grads, _ = tf.clip_by_global_norm(grads, max_norm)
            gpu_grads.append( list(zip(grads, params)) )
            gpu_ops.append([lm_loss])
    tensors = [tf.concat(op, axis=0) for op in zip(*gpu_ops)]
    grad_var = average_grads(gpu_grads)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.apply_gradients(grad_var, global_step=global_step)
    
    return [train_op]+tensors

class ModelESIM(object):
  def __init__(self, features, labels, training=False):
    len1, len2, s1, s2 = features
    K.set_learning_phase(training)
    
    with tf.variable_scope('model'):
      with tf.device('/cpu:0'):
        embedding = tf.get_variable("word2vec", [n_vocab, embed_dim])
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

pjoin = os.path.join

class DatasetLM(object):
    def __init__(self, dir, mode='train', bptt=90):
        path = os.path.join(dir, '{0}_ids.npy'.format(mode))
        self.array = np.load(path)
        self.n_array = len(self.array)
        self.ibar = None
        self.bptt = bptt

    def maxlen(self):
        return self.bptt + 5*5

    def iter(self, epochs, batch_size):
        maxlen = self.maxlen()
        ebar = trange(epochs, desc='Epoch')
        for e in ebar:
            data = random.shuffle(self.array)
            data = np.concatenate(data)
            n_batch = int(data.shape[0]/batch_size)
            data = np.reshape(data[:n_batch*batch_size], [batch_size, -1])

            idx = 0
            self.ibar = trange(n_batch, desc='Iter')
            for i in self.ibar:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
                x = data[:, idx:idx+seq_len]
                seq_len = min(x.shape[1], seq_len)
                x = np.pad(x, ((0,0), (0, maxlen-seq_len)), 'constant')
                yield x, seq_len
    
    def set_postfix(self, **kwargs):
        self.ibar.set_postfix(**kwargs)


def train_lm(datas, holders, train_op, fetchs, epochs=20, batch_size=64):
    data_x, data_l = datas

    n = len(data_l)
    X, L = holders
    lm_loss, = fetchs

    var_list = find_trainable_variables('model')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    res_list = [0. for _ in range(epochs)]
    n_update = len(range(0, n, batch_size))

    try:
        with trange(epochs, desc='Epoch') as ebar:
            for e in ebar:
                data_x, data_l = shuffle(data_x, data_l)
                with trange(0, n, batch_size, desc='Iter') as ibar:
                    for i in ibar:
                        x = data_x[i:i+batch_size]
                        l = data_l[i:i+batch_size]
                        _, loss = sess.run([train_op, lm_loss], feed_dict={X:x, L:l})
                        res_list[e] += loss
                        ibar.set_postfix(loss=loss)
                res_list[e] /= n_update
    finally:
        save_params(save_dir, 'lm_params', sess, var_list)
        for i, res in enumerate(res_list):
            print('{0}\t{1}'.format(i, res))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='process/')
    # parser.add_argument('--maxlen_lm', type=int, default=150)
    parser.add_argument('--maxlen_cl', type=int, default=87) # without delimiter
    parser.add_argument('--n_vocab', type=int, default=3005)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_embd', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)
    random.seed(seed)         # global var seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # load data
    n_position = max(maxlen_lm, maxlen_cl)

    if pretrain:
        lm_dir = pjoin(data_dir, 'lm')
        # trn_lm_x, trn_lm_len = dataset_lm(lm_dir, mode='train')
        # val_lm_x, val_lm_len = dataset_lm(lm_dir, mode='test')
        # print('load data done.')
        trn_lm = DatasetLM(lm_dir)

        X = tf.placeholder(tf.int32, [None, trn_lm.maxlen()])
        L = tf.placeholder(tf.int32, [])

        train_and_tensors = mgpu_train_op_lm((X,L))
        print('build graph done.')
        sys.stdout.flush()

        train_lm((trn_lm_x, trn_lm_len), (X, L), train_op, (lm_loss,), epochs, batch_size)
    else:
        cl_dir = pjoin(data_dir, 'cl')
        if not eval:
            trn_cl_x, trn_cl_len, trn_cl_y = dataset_cl(cl_dir, mode='train')
        val_cl_x, val_cl_len, val_cl_y = dataset_cl(cl_dir, mode='test')

        X = tf.placeholder(tf.int32, [None, 2, maxlen_cl+1, 2])
        L = tf.placeholder(tf.int32, [None])
        Y = tf.placeholder(tf.int32, [None])

        logits, clf_loss, pred, prob, acc = classifier_model(X, L, Y, train=True, reuse=False)
        
        if not eval:
            params = find_trainable_variables("model")
            grads = tf.gradients(clf_loss, params)

            n_update = len(range(0, len(trn_cl_len), batch_size))
            n_updates_total = n_update * epochs

            lr_schedule_fn = partial(warmup_linear, warmup=lr_warmup)
            train_op = adam(params, grads, lr, lr_schedule_fn, n_updates_total, \
                        l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2)
        
            train_cl( (trn_cl_x, trn_cl_len, trn_cl_y), (X,L,Y), train_op, \
                    (clf_loss, acc), epochs, batch_size)
        eval_cl((val_cl_x, val_cl_len, val_cl_y), (X,L,Y), (pred, ))
