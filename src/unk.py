# encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import os
import codecs

vocab = set()
with codecs.open('vocab.all', 'r', 'utf8') as f:
  for line in f:
    vocab.add(line.strip())

with codecs.open('process/vocab.freq', 'r', 'utf8') as f:
  for line in f:
    w = line.split()[0]
    if w not in vocab:
      print(w.encode('utf8'))

