# encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import jieba
import codecs
import os
import re

'''Number of positive and negtive labels
'''

# orig_atec_train = "data/atec/atec_nlp_sim_train.csv"
orig_atec_train_list = ["data/atec/atec_nlp_sim_train.csv",
                        "data/atec/atec_nlp_sim_train_add.csv"]
orig_ccks_train = "data/ccks/train.txt"


atec_freq = {'0': 0, '1': 0}
for orig_atec_train in orig_atec_train_list:
  with codecs.open(orig_atec_train, 'r', 'utf8') as f:
    for i, line in enumerate(f):
      parts = line.strip().lower().split('\t')
      label = parts[3]
      atec_freq[label] += 1
    
ccks_freq = {'0': 0, '1': 0}
with codecs.open(orig_ccks_train, 'r', 'utf8') as f:
  for i, line in enumerate(f):
    parts = line.strip().lower().split('\t')
    label = parts[2]
    ccks_freq[label] += 1

print('atec 0: %d 1: %d' % (atec_freq['0'], atec_freq['1']))
print('ccks 0: %d 1: %d' % (ccks_freq['0'], ccks_freq['1']))

# atec 0: 83792 1: 18685
# ccks 0: 50000 1: 50000
