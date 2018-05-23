# encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import codecs
import os
import random
import time
import math
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from sklearn.metrics import precision_score, recall_score, accuracy_score

from model_kerasqqp import ModelKerasQQP
from model_sialstm import ModelSiameseLSTM
from model_siacnn import ModelSiameseCNN
from model_esim import ModelESIM
from model_bimpm import ModelBiMPM
from model_rnet import ModelRNet

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="bimpm, sialstm, siacnn")
parser.add_argument("--model_name", default="", help="default same as --model")
parser.add_argument("--mode", default="train", help="train, test, debug")
parser.add_argument('--tfdbg', action='store_true', help="debug estimator")
parser.add_argument('--ccks_joint', action='store_true', help="")
parser.add_argument('--ccks_pre', action='store_true', help="")
parser.add_argument('--train_dev', action='store_true', help="")
parser.add_argument('--tune_word', action='store_true', help="")
parser.add_argument("--gpu", default='0', help="")
parser.add_argument("--epochs", default=20, type=int, help="")
parser.add_argument("--start_step", default=0, type=int, help="")
# parser.add_argument("--eval_minutes", default=5, type=int, help="valid in train_eval mode")

args = parser.parse_args()

# Model class
models = {
  "kerasqqp":ModelKerasQQP,
  "sialstm":ModelSiameseLSTM,
  "siacnn":ModelSiameseCNN,
  "esim":ModelESIM,
  "bimpm":ModelBiMPM,
  "rnet": ModelRNet
}

if args.model in models :
  Model = models[args.model]
elif args.model == "debug":
  pass
else:
  raise Exception('model not in %s' % ' '.join(models.keys()))

# files
if args.model_name != "":
  model_dir = "saved_models/model-%s/" % args.model_name
  log_dir = "%s.log" % args.model_name
else:
  model_dir = "saved_models/model-%s/" % args.model
  log_dir = "%s.log" % args.model

out_dir = "process"
vocab_file = out_dir + "/vocab.txt"
embed_file = out_dir + "/embed.npy"

atec_records_basename = out_dir+'/atec_%s.%d.tfrecords'
ccks_records_basename = out_dir+'/ccks_%s.%d.tfrecords'
atec_records = [atec_records_basename % ('train', i) for i in range(4)]
ccks_records = [ccks_records_basename % ('train', i) for i in range(10)]
train_records = []
dev_records = [atec_records_basename % ('dev', 0)]

# runtime hyper parameters

n_instance = 0 # num training instance

if args.ccks_pre:
  n_instance += 10*10000
  train_records += ccks_records

if args.ccks_joint:
  n_instance += 10*10000
  train_records += ccks_records
  n_instance += 39346 - 5000
  train_records += atec_records

if not args.ccks_pre and not args.ccks_joint :
  n_instance += 39346 - 5000
  train_records += atec_records

if args.train_dev:
  n_instance += 5000
  train_records += dev_records


BATCH_SIZE = 32
MAX_STEPS = math.ceil(n_instance * args.epochs / BATCH_SIZE)
if args.start_step !=0:
  MAX_STEPS += args.start_step
LOG_N_ITER = 100
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def get_params():
  return {
        "learning_rate": 0.001, 
        "embed_dim": 256,
        "hidden_size": 200, 
        "dropout": 0.1,
        "rnn_dropout":0.5,
        "max_norm": 5.0,
        "tune_word": args.tune_word,
        # "num_filters" : 230,
        # "kernel_size" : 3,
        # "num_rels" : 53,
        "l2_coef" : 1e-4,
  }

def load_vocab():
  vocab = []
  vocab2id = {}
  with codecs.open(vocab_file, 'r', 'utf8') as f:
    for id, line in enumerate(f):
      token = line.strip()
      vocab.append(token)
      vocab2id[token] = id

  tf.logging.info("load vocab, size: %d" % len(vocab))
  return vocab, vocab2id

def _parse_example(example_proto):
  context_features = {
            'len1': tf.FixedLenFeature([], tf.int64),
            'len2': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)}
  sequence_features = {
            "s1": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "s2": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            }
  context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                                  serialized=example_proto,
                                  context_features=context_features,
                                  sequence_features=sequence_features)

  len1 = context_parsed['len1']
  len2 = context_parsed['len2']
  label = context_parsed['label']

  # s1 = tf.sparse_tensor_to_dense(sequence_parsed['s1'])
  # s2 = tf.sparse_tensor_to_dense(sequence_parsed['s2'])

  s1 = sequence_parsed['s1']
  s2 = sequence_parsed['s2']

  feature = len1, len2, s1, s2
  return feature, label

def _input_fn(filenames, epochs, batch_size, shuffle=False):
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_parse_example)  # Parse the record into tensors.
  if shuffle:
    dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.repeat(epochs)  
  # dataset = dataset.batch(batch_size)
  dataset = dataset.padded_batch(batch_size, (([], [], [None], [None]), []))
  #iterator = dataset.make_initializable_iterator()
  #batch_data = iterator.get_next()
  return dataset

def train_input_fn():
  """An input function for training"""
  return _input_fn(train_records, args.epochs, BATCH_SIZE, shuffle=True)

def dev_input_fn():
  return _input_fn(dev_records, 1, BATCH_SIZE, shuffle=False)

def debug_inputs():
  vocab, _ = load_vocab()
  # dataset = test_input_fn()
  dataset = train_input_fn()
  iter = dataset.make_one_shot_iterator()
  batch_data = iter.get_next()

  with tf.train.MonitoredTrainingSession() as sess:
    # feature, label = sess.run(batch_data)
    # len1, len2, s1, s2 = feature
    # tmp = []
    # for x in s1[0]:
    #   tmp.append(vocab[x])
    # print(' '.join(tmp))
    i = 0
    while not sess.should_stop():
      feature, label = sess.run(batch_data)
      i += 1
      if i % 10000==0:
        print(i)
  print(i)
  # iter 50 for test
  # iter 1344 for train

def debug_model():
  dataset = dev_input_fn()
  iter = dataset.make_one_shot_iterator()
  features, labels = iter.get_next()

  word_embed = np.load(embed_file)
  m = Model(get_params(), word_embed, features, labels, True)

  for v in tf.trainable_variables():
    print(v.name)

  with tf.train.MonitoredTrainingSession() as sess:
    
    for gt in m.gradients:
        print(gt.name)
        
    while not sess.should_stop():
      # _, loss, acc = sess.run([m.train_op, m.loss, m.acc])
      # print(loss, acc)
      
      
      gs = sess.run(m.gradients)
      for g, gt in zip(gs, m.gradients):
        if isinstance(g, np.ndarray):
          try:
            print(gt.name, g.shape)
          except:
            print(gt.name)
            exit()

      # loss, acc = sess.run([m.loss, m.acc])
      # print(loss, acc)

class F1Hook(tf.train.SessionRunHook):
  def __init__(self, pred_tensor, labels_tensor):
    self.pred_tensor = pred_tensor
    self.labels_tensor = labels_tensor
    self.all_pred=[]
    self.all_labels = []
  
  def before_run(self, run_context):
    return tf.train.SessionRunArgs([self.pred_tensor, self.labels_tensor])
  
  def after_run(self, run_context, run_values):
    prob, label = run_values.results
    self.all_pred.append(prob)
    self.all_labels.append(label)

  def end(self, session):
    y_pred = np.concatenate(self.all_pred, axis=0)
    y_true = np.concatenate(self.all_labels,axis=0)
    y_pred = np.reshape(y_pred, [-1])
    y_true = np.reshape(y_true, [-1])

    # np.save('prob.npy', all_prob)
    # np.save('labels.npy', all_labels)
    # tf.logging.info('save results to .npy file')
    
    tf.logging.info("=" * 40)
    acc = accuracy_score(y_true, y_pred)
    tf.logging.info("|| acc: %.3f" % acc)
    
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = 2 * (p * r) / (p + r)
    tf.logging.info("|| binary p: %.3f r: %.3f f1: %.3f" % (p, r, f1))

    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    f1 = 2 * (p * r) / (p + r)
    tf.logging.info("|| macro  p: %.3f r: %.3f f1: %.3f" % (p, r, f1))
    tf.logging.info("=" * 40)


def my_model(features, labels, mode, params):
  """DNN with three hidden layers, and dropout of 0.1 probability."""
  word_embed = np.load(embed_file)

  training = mode == tf.estimator.ModeKeys.TRAIN
  m = Model(params, word_embed, features, labels, training)
  
  # Compute evaluation metrics.
  metrics = {'accuracy': m.acc}
  tf.summary.scalar('accuracy', m.acc[1])

  f1_hook = F1Hook(m.pred,  labels)
  if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=m.loss, eval_metric_ops=metrics, evaluation_hooks=[f1_hook])

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN

  training_hooks = []
  logging_hook = tf.train.LoggingTensorHook({"loss" : m.loss, 
              "accuracy" : m.acc[0]}, 
               every_n_iter=LOG_N_ITER)
  training_hooks.append(logging_hook)
  if args.tfdbg:
    training_hooks.append(tf_debug.LocalCLIDebugHook())
  
  return tf.estimator.EstimatorSpec(mode, loss=m.loss, train_op=m.train_op, 
                training_hooks = training_hooks)

def main(_):
  tf.logging.info('max steps: %d' % MAX_STEPS)
  start_time = time.time()
  if args.model == 'debug':
    debug_inputs()
  else:
    params = get_params()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True # pylint: disable 
    classifier = tf.estimator.Estimator(
          model_fn=my_model,
          model_dir=model_dir,
          config=tf.estimator.RunConfig(session_config=sess_config),
          params=params)
    
    if args.mode == "train":
      classifier.train(input_fn=train_input_fn)
      classifier.evaluate(input_fn=dev_input_fn)
    # elif args.mode == "train_eval" or args.mode == "pretrain":
    #   train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=MAX_STEPS)
    #   eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, 
    #                              steps=None, throttle_secs=60*args.eval_minutes)
    #   tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    elif args.mode == "debug":
      debug_model()
    else:
      eval_result = classifier.evaluate(input_fn=dev_input_fn)
      tf.logging.info('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
  duration = time.time() - start_time
  tf.logging.info("elapsed: %.2f hours" % (duration/3600))
  
    
if __name__=='__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  log = logging.getLogger('tensorflow')
  fh = logging.FileHandler(log_dir)
  log.addHandler(fh)
  tf.app.run()
