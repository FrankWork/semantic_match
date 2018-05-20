# encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import jieba
import codecs
import os
import re

orig_atec_train = "data/atec/atec_nlp_sim_train.csv"
out_dir = "process"
atec_train = out_dir + "/atec_train.tokenized"

orig_ccks_train = "data/ccks/train.txt"
ccks_train = out_dir + "/ccks_train.tokenized"
vocab_freq_file = out_dir + "/vocab.freq"

if not os.path.exists(out_dir):
  os.makedirs(out_dir)

jieba.load_userdict("data/new_words.txt")

# wrong spell mapping data
tokens_map = {}
with codecs.open("data/wrong_words.txt", 'r', 'utf8') as f:
  for line in f:
    parts = line.strip().split()
    wrong_w, correct_w = parts
    tokens_map[wrong_w] = correct_w

def clean_str(string):
  string = re.sub("\*+", "1", string)
  string = re.sub("\d+", "1", string)
  string = re.sub("[一二三四五六七八九]+", "一", string)
  string = re.sub("第一", "第一 ", string)
  for wrong_w in tokens_map:
    correct_w = tokens_map[wrong_w]
    # re.sub(wrong_w, correct_w, string)
    string = string.replace(wrong_w, correct_w)
  
  return string

vocab_freq = {}
f_out = codecs.open(atec_train, 'w', 'utf8')
with codecs.open(orig_atec_train, 'r', 'utf8') as f:
  for i, line in enumerate(f):
    if i!=0 and i%10000 == 0:
      print(i)
    parts = line.strip().lower().split('\t')
    s1, s2, label =parts[1],parts[2],parts[3]
    s1 = clean_str(s1)
    s2 = clean_str(s2)
    s1_toks = [w for w in jieba.cut(s1, HMM=False) if w and w!=' ']
    s2_toks = [w for w in jieba.cut(s2, HMM=False) if w and w!=' ']
    f_out.write("%s\t%s\t%s\n" % (" ".join(s1_toks), " ".join(s2_toks), label))
    for tok in s1_toks + s2_toks:
      if tok not in vocab_freq:
        vocab_freq[tok]=0
      vocab_freq[tok]+=1
f_out.close()

f_out = codecs.open(ccks_train, 'w', 'utf8')
with codecs.open(orig_ccks_train, 'r', 'utf8') as f:
  for i, line in enumerate(f):
    if i!=0 and i%10000 == 0:
      print(i)
    parts = line.strip().lower().split('\t')
    s1, s2, label =parts[0],parts[1],parts[2]
    s1 = clean_str(s1)
    s2 = clean_str(s2)
    s1_toks = [w for w in jieba.cut(s1, HMM=False) if w and w!=' ']
    s2_toks = [w for w in jieba.cut(s2, HMM=False) if w and w!=' ']
    f_out.write("%s\t%s\t%s\n" % (" ".join(s1_toks), " ".join(s2_toks), label))
    for tok in s1_toks + s2_toks:
      if tok not in vocab_freq:
        vocab_freq[tok]=0
      vocab_freq[tok]+=1
f_out.close()

vocab_freq =[(tok, vocab_freq[tok]) for tok in vocab_freq.keys()]
vocab_freq = sorted(vocab_freq, key=lambda tup: tup[1])

with codecs.open(vocab_freq_file, 'w', 'utf8') as f_out:
  for k,v in vocab_freq:
    f_out.write("%s\t%d\n" %(k,v))

    
