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
import numpy as np
import tensorflow as tf
from model_bimpm import ModelBiMPM


out_dir = "process"
vocab_file = out_dir + "/vocab.txt"
embed_file = out_dir + "/embed.npy"

atec_records_basename = out_dir+'/atec_%s.%d.tfrecords'
ccks_records_basename = out_dir+'/ccks_%s.%d.tfrecords'

model_dir = "saved_models/model-bimpm/"

BATCH_SIZE = 100
EPOCHS = 20
LOG_N_ITER = 10

def get_params():
  return {
        "learning_rate": 0.001, 
        "embed_dim": 256,
        "hidden_size": 200, 
        "dropout": 0.1,
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
  train_filenames = atec_files
  return _input_fn(train_filenames, EPOCHS, BATCH_SIZE, shuffle=True)

def test_input_fn():
  test_filenames = [atec_records_basename % ('dev', 0)]
  return _input_fn(test_filenames, 1, BATCH_SIZE, shuffle=False)

def debug_inputs():
  vocab, _ = load_vocab()
  dataset = test_input_fn()
  iter = dataset.make_one_shot_iterator()
  batch_data = iter.get_next()

  with tf.train.MonitoredTrainingSession() as sess:
    # feature, label = sess.run(batch_data)
    # len1, len2, s1, s2 = feature
    # tmp = []
    # for x in s1[0]:
    #   tmp.append(vocab[x])
    # print(' '.join(tmp))
    while not sess.should_stop():
      feature, label = sess.run(batch_data)
      len1, len2, s1, s2 = feature
      print(s1, s2)

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

def my_model(features, labels, mode, params):
  """DNN with three hidden layers, and dropout of 0.1 probability."""
  word_embed = np.load(embed_file)

  training = mode == tf.estimator.ModeKeys.TRAIN
  m = ModelBiMPM(params, word_embed, features, labels, training)
  
  # Compute evaluation metrics.
  metrics = {'accuracy': m.acc}
  tf.summary.scalar('accuracy', m.acc[1])

  if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=m.loss, eval_metric_ops=metrics, evaluation_hooks=[])

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN

  logging_hook = tf.train.LoggingTensorHook({"loss" : m.loss, 
              "accuracy" : m.acc[0]}, 
               every_n_iter=LOG_N_ITER)
  
  return tf.estimator.EstimatorSpec(mode, loss=m.loss, train_op=m.train_op, 
                training_hooks = [logging_hook])


class PatTopKHook(tf.train.SessionRunHook):
  def __init__(self, prob_tensor, labels_tensor):
    self.prob_tensor = prob_tensor
    self.labels_tensor = labels_tensor
    self.all_prob=[]
    self.all_labels = []
  
  def before_run(self, run_context):
    return tf.train.SessionRunArgs([self.prob_tensor, self.labels_tensor])
  
  def after_run(self, run_context, run_values):
    prob, label = run_values.results
    self.all_prob.append(prob)
    self.all_labels.append(label)

  def end(self, session):
    all_prob = np.concatenate(self.all_prob, axis=0)
    all_labels = np.concatenate(self.all_labels,axis=0)

    np.save('prob.npy', all_prob)
    np.save('labels.npy', all_labels)
    tf.logging.info('save results to .npy file')
    
    bag_size, num_class = all_prob.shape
    mask = np.ones([num_class])
    mask[0]=0
    mask_prob = np.reshape(all_prob*mask, [-1])
    idx_prob = mask_prob.argsort()

    one_hot_labels = np.zeros([bag_size, num_class])
    one_hot_labels[np.arange(bag_size), all_labels] = 1
    one_hot_labels = np.reshape(one_hot_labels, [-1])

    idx = idx_prob[-100:][::-1]
    p100 = np.mean(one_hot_labels[idx])
    idx = idx_prob[-200:][::-1]
    p200 = np.mean(one_hot_labels[idx])
    idx = idx_prob[-500:][::-1]
    p500 = np.mean(one_hot_labels[idx])

    tf.logging.info("p@100: %.3f p@200: %.3f p@500: %.3f" % (p100, p200, p500))
    tf.logging.info(all_prob[-1][:5])

def main(_):
  # if not os.path.exists(FLAGS.out_dir):
  #   os.makedirs(FLAGS.out_dir)
  
  start_time = time.time()
  params = get_params()
  classifier = tf.estimator.Estimator(
        model_fn=my_model,
        model_dir=model_dir,
        params=params)
  classifier.train(input_fn=train_input_fn)

  eval_result = classifier.evaluate(input_fn=test_input_fn)
  tf.logging.info('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
  duration = time.time() - start_time
  tf.logging.info("elapsed: %.2f hours" % (duration/3600))

  # debug_inputs()
  # debug_model()
    
if __name__=='__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  # log = logging.getLogger('tensorflow')
  # fh = logging.FileHandler('tmp.log')
  # log.addHandler(fh)
  tf.app.run()
