# encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import jieba
import codecs
import os
import numpy as np

out_dir = "process"
vocab_freq_file = out_dir + "/vocab.freq"
vocab_file = out_dir + "/vocab.txt"
embed_file = out_dir + "/embed.npy"

orig_embed_dir = os.environ['HOME']+"/work/data/word_embedding/embedding/embedding.giga/embedding.iter.50"
orig_embed_file= "all.vector.skip.256"
min_freq = 1
embed_dim = 256

vocab_freq = {}
with codecs.open(vocab_freq_file, 'r', 'utf8') as f:
  for line in f:
    parts = line.strip().split()
    if len(parts) != 2:
      continue
    tok, freq = parts[0], parts[1]
    if freq > min_freq:
      vocab_freq[tok] = freq
print('load total %d tokens' % len(vocab_freq))

orig_embed_path = os.path.join(orig_embed_dir, orig_embed_file)
vocab = ['PAD', 'UNK']
embed = [np.zeros([embed_dim]), np.random.normal(0,0.1,[embed_dim])]
with codecs.open(orig_embed_path, 'r', 'gbk') as f:
  for i, line in enumerate(f):
    if i==0:  # vocab number
      continue
    parts = line.strip().split()
    tok = parts[0]
    if tok in vocab_freq:
      vec = [float(x) for x in parts[1:]]
      assert len(vec) == embed_dim
      vocab.append(tok)
      embed.append(vec)
print('load %d tokens with pretrained embedding' % len(vocab))

with codecs.open(vocab_file, 'w', 'utf8') as f:
  for tok in vocab:
    f.write("%s\n" % tok)

embed = np.asarray(embed)
np.save(embed_file, embed.astype(np.float32))