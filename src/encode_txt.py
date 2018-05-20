# encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import jieba
import codecs
import os



out_dir = "process"
vocab_file = out_dir + "/vocab.txt"
atec_train = out_dir + "/atec_train.tokenized"

def load_vocab():
  vocab = []
  vocab2id = {}
  with codecs.open(vocab_file, 'r', 'utf8') as f:
    for id, line in enumerate(f):
      token = line.strip()
      vocab.append(token)
      vocab2id[token] = id

  return vocab, vocab2id

_, vocab_idx = load_vocab()
fout = codecs.open('tmp.txt', 'w', 'utf8')
with codecs.open(atec_train, 'r', 'utf8') as fin:
  for line in fin:
    parts = line.strip().split('\t')
    s1,s2,label = parts[0],parts[1],parts[2]

    s1 = [tok if tok in vocab_idx else 'UNK'
                          for tok in s1.split() ]
    s2 = [tok if tok in vocab_idx else 'UNK'
                          for tok in s2.split() ]
    fout.write('%s %s %s\n' % (" ".join(s1), ' '.join(s2), label))
fout.close()
