# encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import jieba
import codecs
import os
import re
import argparse
import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("--infile", help="")
parser.add_argument("--outfile")
args = parser.parse_args()

feature = tf.train.Feature
sequence_example = tf.train.SequenceExample

def features(d): return tf.train.Features(feature=d)
def int64_feature(v): return feature(int64_list=tf.train.Int64List(value=v))
def bytes_feature(v): return feature(bytes_list=tf.train.BytesList(value=v))
def feature_list(l): return tf.train.FeatureList(feature=l)
def feature_lists(d): return tf.train.FeatureLists(feature_list=d)


jieba.load_userdict("new_words.txt")

# wrong spell mapping data
tokens_map = {}
with codecs.open("wrong_words.txt", 'r', 'utf8') as f:
  for line in f:
    parts = line.strip().split()
    wrong_w, correct_w = parts
    tokens_map[wrong_w] = correct_w

vocab_idx = {}
with codecs.open('vocab.txt', 'r', 'utf8') as f:
  for idx, line in enumerate(f):
    tok = line.strip()
    vocab_idx[tok] = idx

def clean_str(string):
  string = re.sub("\*\*\*", "1", string)
  string = re.sub("\d+", "1", string)
  string = re.sub("[一二三四五六七八九十百千万两]+", "一 ", string)
  string = re.sub("第一", "第一 ", string)
  for wrong_w in tokens_map:
    correct_w = tokens_map[wrong_w]
    # re.sub(wrong_w, correct_w, string)
    string = string.replace(wrong_w, correct_w)
  
  return string


def convert_example(s1, s2):
  s1 = [vocab_idx[tok] if tok in vocab_idx else vocab_idx['UNK']
                         for tok in s1]
  s2 = [vocab_idx[tok] if tok in vocab_idx else vocab_idx['UNK']
                         for tok in s2]
  len1, len2 = len(s1), len(s2)
  s1 = [int64_feature([x]) for x in s1]
  s2 = [int64_feature([x]) for x in s2]
  
  example = sequence_example(
                  context=features({
                    'label': int64_feature([0]),
                    'len1': int64_feature([len1]),
                    'len2': int64_feature([len2]),
                  }),
                  feature_lists=feature_lists({
                      "s1": feature_list(s1),
                      "s2": feature_list(s2),
                  }))
  return example


writer = tf.python_io.TFRecordWriter(args.outfile)
with codecs.open(args.infile, 'r', 'utf8') as f:
  for i, line in enumerate(f):
    if i!=0 and i%10000 == 0:
      print(i)
    parts = line.strip().lower().split('\t')
    s1, s2 =parts[1],parts[2]
    s1 = clean_str(s1)
    s2 = clean_str(s2)
    s1_toks = [w for w in jieba.cut(s1, HMM=False) if w and w!=' ']
    s2_toks = [w for w in jieba.cut(s2, HMM=False) if w and w!=' ']

    example = convert_example(s1_toks, s2_toks)
    writer.write(example.SerializeToString())
writer.close()