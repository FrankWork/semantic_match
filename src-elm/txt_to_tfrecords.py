# encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import jieba
import codecs
import os
import random
import numpy as np
import tensorflow as tf


out_dir = "process"
vocab_file = out_dir + "/vocab.txt"
embed_file = out_dir + "/embed.npy"

atec_train = out_dir + "/atec_train.tokenized"
ccks_train = out_dir + "/ccks_train.tokenized"

atec_records_basename = out_dir+'/atec_%s.%d.tfrecords'
ccks_records_basename = out_dir+'/ccks_%s.%d.tfrecords'

NUM_DEV = 10000
MAX_INSTANCE = 10000

vocab_idx = {}
with codecs.open(vocab_file, 'r', 'utf8') as f:
  for idx, line in enumerate(f):
    tok = line.strip()
    vocab_idx[tok] = idx



feature = tf.train.Feature
sequence_example = tf.train.SequenceExample

def features(d): return tf.train.Features(feature=d)
def int64_feature(v): return feature(int64_list=tf.train.Int64List(value=v))
def bytes_feature(v): return feature(bytes_list=tf.train.BytesList(value=v))
def feature_list(l): return tf.train.FeatureList(feature=l)
def feature_lists(d): return tf.train.FeatureLists(feature_list=d)


def convert_example(line):
  parts = line.strip().split('\t')
  s1,s2,label = parts[0],parts[1],parts[2]
  label = int(label)

  s1 = [vocab_idx[tok] if tok in vocab_idx else vocab_idx['UNK']
                         for tok in s1.split() ]
  s2 = [vocab_idx[tok] if tok in vocab_idx else vocab_idx['UNK']
                         for tok in s2.split() ]
  len1, len2 = len(s1), len(s2)
  s1 = [int64_feature([x]) for x in s1]
  s2 = [int64_feature([x]) for x in s2]
  
  example = sequence_example(
                  context=features({
                    'label': int64_feature([label]),
                    'len1': int64_feature([len1]),
                    'len2': int64_feature([len2]),
                  }),
                  feature_lists=feature_lists({
                      "s1": feature_list(s1),
                      "s2": feature_list(s2),
                  }))
  return example

def write_tfrecords(data, records_basename, mode, max_instance=MAX_INSTANCE):
  if mode=='train':
    print('convert train set')
    file_id = 0
    for i in range(0, len(data), MAX_INSTANCE):
      file = records_basename %  (mode, file_id)
      file_id+=1
      with tf.python_io.TFRecordWriter(file) as writer:
        for line in data[i:i+MAX_INSTANCE]:
          example = convert_example(line)
          writer.write(example.SerializeToString())
      print('convert %d instance' % (i+MAX_INSTANCE) )
  else:
    print('convert dev set')
    file = records_basename % (mode, 0)
    with tf.python_io.TFRecordWriter(file) as writer:
      for line in data:
        example = convert_example(line)
        writer.write(example.SerializeToString())


def convert_tfrecords(txt_file, records_basename, num_dev=NUM_DEV, shuffle=True):
  print('convert %s to tfrecords' % os.path.basename(txt_file))
  with codecs.open(txt_file, 'r', 'utf8') as f:
    lines = f.readlines()
  
  if shuffle:
    random.seed(41)
    random.shuffle(lines)

  if num_dev != 0:
    train_set = lines[:-num_dev]
    dev_set = lines[-num_dev:]
    write_tfrecords(train_set, records_basename, "train")
    write_tfrecords(dev_set, records_basename, "dev")
  else:
    train_set = lines
    write_tfrecords(train_set, records_basename, "train")

    
convert_tfrecords(atec_train, atec_records_basename)
convert_tfrecords(ccks_train, ccks_records_basename, num_dev=0)
