# encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import jieba
import codecs
import os
import random
import time
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from model_kerasqqp import ModelKerasQQP
from model_sialstm import ModelSiameseLSTM
from model_siacnn import ModelSiameseCNN
from model_esim import ModelESIM
from model_bimpm import ModelBiMPM

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="bimpm, sialstm, siacnn, debug")
parser.add_argument("--mode", default="train", help="train, test")
parser.add_argument("--epochs", default=40, help="", type=int)
parser.add_argument("--gpu", default=0, help="")
args = parser.parse_args()

# export CUDA_VISIBLE_DEVICES=2
# python src/estimator.py --model bimpm
# python src/estimator.py --model sialstm

models = {
  "kerasqqp":ModelKerasQQP,
  "sialstm":ModelSiameseLSTM,
  "siacnn":ModelSiameseCNN,
  "esim":ModelESIM,
  "bimpm":ModelBiMPM
}

if args.model in models :
  Model = models[args.model]
elif args.model == "debug":
  pass
else:
  raise Exception('model not in %s' % ' '.join(models.keys()))

model_dir = "saved_models/model-%s/" % args.model

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

out_dir = "process"
vocab_file = out_dir + "/vocab.txt"
embed_file = out_dir + "/embed.npy"

atec_records_basename = out_dir+'/atec_%s.%d.tfrecords'
ccks_records_basename = out_dir+'/ccks_%s.%d.tfrecords'

# CUDA_VISIBLE_DEVICES=2
BATCH_SIZE = 100
LOG_N_ITER = 100

def get_params():
  return {
        "learning_rate": 0.001, 
        "embed_dim": 256,
        "hidden_size": 200, 
        "dropout": 0.1,
        "rnn_dropout":0.5,
        "max_norm": 5.0,
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
  atec_files = [atec_records_basename % ('train', i) for i in range(4)]
  ccks_files = [ccks_records_basename % ('train', i) for i in range(10)]
  train_filenames = atec_files + ccks_files
  return _input_fn(train_filenames, args.epochs, BATCH_SIZE, shuffle=True)

def test_input_fn():
  test_filenames = [atec_records_basename % ('dev', 0)]
  return _input_fn(test_filenames, 1, BATCH_SIZE, shuffle=False)

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
  dataset = test_input_fn()
  iter = dataset.make_one_shot_iterator()
  features, labels = iter.get_next()

  word_embed = np.load(embed_file)
  m = ModelBiMPM(get_params(), word_embed, features, labels, False)

  with tf.train.MonitoredTrainingSession() as sess:
    prob, pred = sess.run([m.prob, m.pred])
    print(prob)
    print(pred)

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
    all_pred = np.concatenate(self.all_pred, axis=0)
    all_labels = np.concatenate(self.all_labels,axis=0)
    all_pred = np.reshape(all_pred, [-1])
    all_labels = np.reshape(all_labels, [-1])
    f1 = f1_score(all_labels, all_pred, average='macro')
    # np.save('prob.npy', all_prob)
    # np.save('labels.npy', all_labels)
    # tf.logging.info('save results to .npy file')
    tf.logging.info("f1: %.3f" % f1)

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

  logging_hook = tf.train.LoggingTensorHook({"loss" : m.loss, 
              "accuracy" : m.acc[0]}, 
               every_n_iter=LOG_N_ITER)
  
  return tf.estimator.EstimatorSpec(mode, loss=m.loss, train_op=m.train_op, 
                training_hooks = [logging_hook])

def main(_):
  # if not os.path.exists(FLAGS.out_dir):
  #   os.makedirs(FLAGS.out_dir)
  
  start_time = time.time()
  if args.model == 'debug':
    debug_inputs()
  # debug_model()
  else:
    params = get_params()
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth=True

    classifier = tf.estimator.Estimator(
          model_fn=my_model,
          model_dir=model_dir,
          params=params)
    
    if args.mode == "train":
      classifier.train(input_fn=train_input_fn)

    eval_result = classifier.evaluate(input_fn=test_input_fn)
    tf.logging.info('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
  duration = time.time() - start_time
  tf.logging.info("elapsed: %.2f hours" % (duration/3600))

  
    
if __name__=='__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  # log = logging.getLogger('tensorflow')
  # fh = logging.FileHandler('tmp.log')
  # log.addHandler(fh)
  tf.app.run()
