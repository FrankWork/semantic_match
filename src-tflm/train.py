from __future__ import division
import os
import sys
import time
import math
import json
import random
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function

from tqdm import tqdm, trange
from functools import partial
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def np_softmax(x, t=1):
    x = x/t
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex/np.sum(ex, axis=-1, keepdims=True)

def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f

def _identity_init(shape, dtype, partition_info, scale):
    n = shape[-1]
    w = np.eye(n)*scale
    if len([s for s in shape if s != 1]) == 2:
        w = w.reshape(shape)
    return w.astype(np.float32)

def identity_init(scale=1.0):
    return partial(_identity_init, scale=scale)

def _np_init(shape, dtype, partition_info, w):
    return w

def np_init(w):
    return partial(_np_init, w=w)

class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log = open(make_path(path), 'w')
        self.f_log.write(json.dumps(kwargs)+'\n')

    def log(self, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log.write(json.dumps(kwargs)+'\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()

def find_trainable_variables(key):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))

def flatten(outer):
    return [el for inner in outer for el in inner]

def remove_none(l):
    return [e for e in l if e is not None]

def get_ema_if_exists(v, gvs):
    name = v.name.split(':')[0]
    ema_name = name+'/ExponentialMovingAverage:0'
    ema_v = [v for v in gvs if v.name == ema_name]
    if len(ema_v) == 0:
        ema_v = [v]
    return ema_v[0]

def get_ema_vars(*vs):
    if tf.get_variable_scope().reuse:
        gvs = tf.global_variables()
        vs = [get_ema_if_exists(v, gvs) for v in vs]
    if len(vs) == 1:
        return vs[0]
    else:
        return vs

@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """force gradient to be a dense tensor
    it's often faster to do dense embedding gradient on GPU than sparse on CPU
    """
    return x

def assign_to_gpu(gpu=0, ps_dev="/device:CPU:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return ps_dev
        else:
            return "/gpu:%d" % gpu
    return _assign

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

def warmup_cosine(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*(0.5 * (1 + tf.cos(math.pi * x)))

def warmup_constant(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*1

def warmup_linear(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return (s*(x/warmup) + (1-s))*(1-x)

def adam(params, grads, lr, schedule, t_total, b1=0.9, b2=0.999, e=1e-8, l2=0, vector_l2=False, max_grad_norm=-1, **kwargs):
    """
    adam with weight decay fix
    """
    t = tf.Variable(0, dtype=tf.float32, trainable=False)
    tt = t+1
    updates = [t.assign(tt)]
    if max_grad_norm > 0:
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    for p, g in zip(params, grads):
        if p is None or g is None:
            print("can't train", p.name, g)
        else:
            if isinstance(g, tf.IndexedSlices):
                g = tf.convert_to_tensor(g)
            m = tf.Variable(p*0, dtype=tf.float32, trainable=False)
            v = tf.Variable(p*0, dtype=tf.float32, trainable=False)
            lrt = lr*tf.sqrt(1-b2**tt)/(1-b1**tt)
            lrt *= schedule(t/t_total)
            mt = b1*m + (1-b1)*g
            vt = b2*v + (1-b2)*g*g
            if (len(p.get_shape()) > 1 or vector_l2) and l2 > 0:
                pt = p - lrt * (mt / (tf.sqrt(vt) + e) + l2*p)
            else:
                pt = p - lrt * (mt / (tf.sqrt(vt) + e))
            updates.extend([m.assign(mt), v.assign(vt), p.assign(pt)])
    return tf.group(*updates)

def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def swish(x):
    return x*tf.nn.sigmoid(x)

act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}

def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        g, b = get_ema_vars(g, b)
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w

def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return a

def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def attn(x, scope, n_state, n_head, train=False, scale=False):
    assert n_state%n_head==0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        a = dropout(a, resid_pdrop, train)
        return a

def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns[afn]
        h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train)
        h2 = dropout(h2, resid_pdrop, train)
        return h2

def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, train=train, scale=scale)
        n = norm(x+a, 'ln_1')
        m = mlp(n, 'mlp', nx*4, train=train)
        h = norm(n+m, 'ln_2')
        return h

def embed(X, we):
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h

def clf(x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w)+b

def language_model(X, L, train=False, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        
        M = tf.to_float(tf.sequence_mask(L, maxlen_lm)) # (n, maxlen)

        we = tf.get_variable("we", [n_vocab+n_position, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)

        h = embed(X, we)
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=train, scale=True)

        lm_h = tf.reshape(h[:, :-1], [-1, n_embd]) # remove the last token of lm_h
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits=lm_logits, 
                                        labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)

        return tf.reduce_mean(lm_losses)

def classifier_model(X, L, Y, train=False, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        
        l2d = tf.tile(tf.expand_dims(L, axis=1), [1, 2])    # (n, 2)
        M = tf.to_float(tf.sequence_mask(l2d, maxlen_cl+1)) # (n, 2, maxlen)

        we = tf.get_variable("we", [n_vocab+n_position, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)

        X = tf.reshape(X, [-1, n_ctx, 2])
        M = tf.reshape(M, [-1, n_ctx])

        h = embed(X, we)
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=train, scale=True)

        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits=lm_logits, 
                                        labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)

        clf_h = tf.reshape(h, [-1, n_embd])
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*n_ctx+pool_idx)

        clf_h = tf.reshape(clf_h, [-1, 2, n_embd])
        if train and clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1-clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, n_embd])
        clf_logits = clf(clf_h, 1, train=train)
        clf_logits = tf.reshape(clf_logits, [-1, 2])

        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
        return clf_logits, clf_losses, lm_losses

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
                        res_list[e] += loss/len(x)
                        ibar.set_postfix(loss=loss)
                res_list[e] /= n_update
    finally:
        save_params(save_dir, 'lm_params', sess, var_list)
        for i, res in enumerate(res_list):
            print('{0}\t{1}'.format(i, res))

def save_params(dir, file, sess, var_list):
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    saver = tf.train.Saver(var_list)
    saver.save(sess, os.path.join(dir, file))

def load_params(dir, file, sess, var_list):
    path = os.path.join(dir, file)
    assert os.path.exists(path), 'path {0} not exists'.format(path)
    
    saver = tf.train.Saver(var_list)
    saver.restore(sess, path)

def log():
    global best_score
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_cost/len(trY[:n_valid])
    va_cost = va_cost/n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1))*100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1))*100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, 
               tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f'%(n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            save_params(dir, 'best_params', sess, var_list)

argmax = lambda x:np.argmax(x, 1)

def predict():
    filename = 'ROCStories.tsv'
    label_decoder = None
    predictions = argmax(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))

pjoin = os.path.join

def dataset_lm(dir, mode='train'):
    path = pjoin(dir, '{0}_ids.npy'.format(mode))
    data = np.load(path)
    n = len(data)

    X = np.zeros((n, maxlen_lm, 2), dtype=np.int32)
    length = np.zeros(n, dtype=np.int32)
    
    for i, x in enumerate(data):
        m = len(x)
        length[i] = m
        X[i, :m, 0] = x
    X[:, :, 1] = np.arange(n_vocab, n_vocab + maxlen_lm)# positional feature
    return X, length

def dataset_cl(dir, mode='train'):
    x_path = pjoin(dir, '{0}_ids.npy'.format(mode))
    y_path = pjoin(dir, '{0}_labels.npy'.format(mode))
    data = np.load(x_path)
    Y = np.load(y_path)
    n = len(data)

    delimiter = [ n_vocab - 1 ]

    X = np.zeros((n, 2, maxlen_cl+1, 2), dtype=np.int32)
    length = np.zeros(n, dtype=np.int32)
    
    for i, x in enumerate(data):
        m = len(x[0]) + len(x[1]) + 1
        length[i] = m
        X[i, 0, :m, 0] = x[0] + delimiter + x[1]
        X[i, 1, :m, 0] = x[1] + delimiter + x[0]
    X[:, :, :, 1] = np.arange(n_vocab, n_vocab + maxlen_cl + 1) # positional feature
    return X, Y, length

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='process/')
    parser.add_argument('--maxlen_lm', type=int, default=155)
    parser.add_argument('--maxlen_cl', type=int, default=167) # without delimiter
    parser.add_argument('--n_vocab', type=int, default=1425)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_embd', type=int, default=300)

    # parser.add_argument('--desc', type=str)
    # parser.add_argument('--dataset', type=str)
    # parser.add_argument('--log_dir', type=str, default='log/')
    # parser.add_argument('--submission_dir', type=str, default='submission/')
    # parser.add_argument('--submit', action='store_true')
    # parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_head', type=int, default=10)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=4)
    parser.add_argument('--afn', type=str, default='gelu')
    # parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    # parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    # parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    # parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    
    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)
    random.seed(seed)         # global var seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # load data
    n_position = max(maxlen_lm, maxlen_cl+1)

    if pretrain:
        lm_dir = pjoin(data_dir, 'lm')
        trn_lm_x, trn_lm_len = dataset_lm(lm_dir, mode='train')
        val_lm_x, val_lm_len = dataset_lm(lm_dir, mode='test')
        print('load data done.')

        X = tf.placeholder(tf.int32, [None, maxlen_lm, 2])
        L = tf.placeholder(tf.int32, [None])

        lm_loss = language_model(X, L, train=True, reuse=False)
        train_op = tf.train.AdamOptimizer().minimize(lm_loss)
        print('build graph done.')
        sys.stdout.flush()

        train_lm((trn_lm_x, trn_lm_len), (X, L), train_op, (lm_loss,), epochs=20, batch_size=64)

    else:
        cl_dir = pjoin(data_dir, 'cl')
        trn_cl_x, trn_cl_y, trn_cl_len = dataset_cl(cl_dir, mode='train')
        val_cl_x, val_cl_y, val_cl_len = dataset_cl(cl_dir, mode='test')

        X = tf.placeholder(tf.int32, [None, 2, maxlen_cl+1, 2])
        L = tf.placeholder(tf.int32, [None])
        Y = tf.placeholder(tf.int32, [None])

        clf_logits, clf_losses, lm_losses = classifier_model(X, L, Y, train=True, reuse=False)
        
        if lm_coef > 0:
            train_loss = tf.reduce_mean(clf_losses) + lm_coef*tf.reduce_mean(lm_losses)
        else:
            train_loss = tf.reduce_mean(clf_losses)
        params = find_trainable_variables("model")
        grads = tf.gradients(train_loss, params)

        lr_schedule_fn = partial(warmup_linear, warmup=lr_warmup)
        train = adam(params, grads, lr, lr_schedule_fn, n_updates_total, \
                    l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2)
        
        load_params(dir, file, sess, var_list)

    