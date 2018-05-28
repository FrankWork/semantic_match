# encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import jieba
import codecs
import os
import time
import numpy as np

out_dir = "process"
vocab_freq_file = out_dir + "/vocab.freq"
vocab_file = out_dir + "/vocab.txt"
embed_file = out_dir + "/embed.npy"

orig_embed_dir = os.environ['HOME']+"/work/data/word_embedding/embedding/embedding.giga/"
orig_embed_files= ["embedding.deprl.basic_split/all_cmn.deprl.256.vecs",
      "embedding.deprl.basic_split/xin_cmn.deprl.256.vecs",
      "embedding.iter.15/all.vector.skip.256",
      "embedding.iter.15/xin.vector.skip.256"]
# 457620
# 162648 
# 742453 
# 242730 

min_freq = 1
embed_dim = 256

t_begin = time.time()

vocab_freq = {}
with codecs.open(vocab_freq_file, 'r', 'utf8') as f:
  for line in f:
    parts = line.strip().split()
    if len(parts) != 2:
      continue
    tok, freq = parts[0], int(parts[1])
    if freq > min_freq:
      vocab_freq[tok] = freq
print('load total %d tokens' % len(vocab_freq))

vocab = ['PAD', 'UNK']
embed = [np.zeros([embed_dim]), np.random.normal(0,0.1,[embed_dim])]
vocab_seen = set()

for orig_embed_file in orig_embed_files:
  orig_embed_path = os.path.join(orig_embed_dir, orig_embed_file)
  
  with codecs.open(orig_embed_path, 'r', 'gbk') as f:
    line = f.readline()
    print(orig_embed_file, line.strip())
    n_error = 0
    n_line = 0
    while True:
      try:
        line = f.readline()
      except UnicodeDecodeError:
        n_error += 1
        continue
      except EOFError:
        break
      parts = line.strip().split()
      n_line += 1
      if n_line % 10000 == 0:
        print(n_line)
      if len(parts) < embed_dim:
        break
      tok = parts[0]
      if tok in vocab_freq and tok not in vocab_seen:
        vec = [float(x) for x in parts[1:]]
        assert len(vec) == embed_dim
        vocab.append(tok)
        vocab_seen.add(tok)
        embed.append(vec)
    print('%d decode error' % n_error)
print('load %d tokens with pretrained embedding' % len(vocab))

n_random = len(vocab_freq) - len(vocab)
for tok in vocab_freq:
  if tok not in vocab_seen:
    vocab.append(tok)
    vec = np.random.normal(0,0.1,[embed_dim])
    vocab_seen.add(tok)
    embed.append(vec)
print('random init %d embeddings' % n_random)

with codecs.open(vocab_file, 'w', 'utf8') as f:
  for tok in vocab:
    f.write("%s\n" % tok)

embed = np.asarray(embed)
np.save(embed_file, embed.astype(np.float32))

t_total = time.time() - t_begin
print('time %d secs' % t_total)