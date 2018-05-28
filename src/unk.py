# encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import os
import time
import codecs

orig_embed_dir = os.environ['HOME']+"/work/data/word_embedding/embedding/embedding.giga/"
orig_embed_files= ["embedding.deprl.basic_split/all_cmn.deprl.256.vecs",
      "embedding.deprl.basic_split/xin_cmn.deprl.256.vecs",
      "embedding.iter.15/all.vector.skip.256",
      "embedding.iter.15/xin.vector.skip.256"]

out_dir = "process"
all_vocab_file = out_dir + "/vocab.all"
vocab_freq_file = out_dir + "/vocab.freq"
embed_dim = 256

t_begin = time.time()

## extrac vocab from pre-trained embedding
# print('extract all pre-trained vocab')
# all_vocab = set()
# for orig_embed_file in orig_embed_files:
#   orig_embed_path = os.path.join(orig_embed_dir, orig_embed_file)
#   with codecs.open(orig_embed_path, 'r', 'gbk') as f:
#     line = f.readline()
#     print(orig_embed_file, line.strip())
#     n_error = 0
#     n_line = 0
#     while True:
#       try:
#         line = f.readline()
#       except UnicodeDecodeError:
#         n_error += 1
#         continue
#       except EOFError:
#         break
#       parts = line.strip().split()
#       n_line += 1
#       if n_line % 10000 == 0:
#         print(n_line)
#       if len(parts) < embed_dim:
#         break
#       tok = parts[0]
#       all_vocab.add(tok)
#     print('%d decode error' % n_error)
# print('load %d tokens with pretrained embedding' % len(all_vocab))

# all_vocab = sorted(list(all_vocab))
# with codecs.open(all_vocab_file, 'w', 'utf8') as f:
#   for tok in all_vocab:
#     f.write('%s\n' % tok)



vocab = set()
with codecs.open(all_vocab_file, 'r', 'utf8') as f:
  for line in f:
    vocab.add(line.strip())

with codecs.open(vocab_freq_file, 'r', 'utf8') as f:
  for line in f:
    w = line.split()[0]
    if w not in vocab:
      print(w)#.encode('utf8')



t_total = time.time() - t_begin
print('time %d secs' % t_total)
