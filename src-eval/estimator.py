# encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import jieba
import codecs
import os
import argparse
import random
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from model_bimpm import ModelBiMPM


parser = argparse.ArgumentParser()
parser.add_argument("--infile", help="")
parser.add_argument("--outfile")
parser.add_argument("--model_dir")
args = parser.parse_args()


embed_file = "embed.npy"

# CUDA_VISIBLE_DEVICES=2
BATCH_SIZE = 100

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

def test_input_fn():
  test_filenames = [args.infile]
  return _input_fn(test_filenames, 1, BATCH_SIZE, shuffle=False)

class WritePredHook(tf.train.SessionRunHook):
  def __init__(self, pred_tensor):
    self.pred_tensor = pred_tensor
    self.all_pred=[]
  
  def before_run(self, run_context):
    return tf.train.SessionRunArgs([self.pred_tensor])
  
  def after_run(self, run_context, run_values):
    prob = run_values.results
    self.all_pred.append(prob)

  def end(self, session):
    all_pred = np.concatenate(self.all_pred, axis=1)
    all_pred = list(np.reshape(all_pred, [-1]))
    with codecs.open(args.outfile, 'w', 'utf8') as f:
      for i in range(len(all_pred)):
        f.write('%d\t%d\n' %(i+1, all_pred[i]))

def my_model(features, labels, mode, params):
  """DNN with three hidden layers, and dropout of 0.1 probability."""
  word_embed = np.load(embed_file)

  training = mode == tf.estimator.ModeKeys.TRAIN
  m = ModelBiMPM(params, word_embed, features, labels, training)
  
  # Compute evaluation metrics.
  metrics = {'accuracy': m.acc}

  write_hook = WritePredHook(m.pred)
  if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=m.loss, eval_metric_ops=metrics, evaluation_hooks=[write_hook])

def main(_):
  params = get_params()
  classifier = tf.estimator.Estimator(
        model_fn=my_model,
        model_dir=args.model_dir,
        params=params)

  _ = classifier.evaluate(input_fn=test_input_fn)
    
if __name__=='__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
