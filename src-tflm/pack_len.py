import numpy as np


class LanguageModelLoader():
    # from `fastai`    
    """ Returns a language model iterator that iterates through batches that are of length N(bptt,5)
    The first batch returned is always bptt+25; the max possible width.  This is done because of they way that pytorch
    allocates cuda memory in order to prevent multiple buffers from being created as the batch width grows.
    """
    def __init__(self, nums, bs, bptt, backwards=False):
        self.bs,self.bptt,self.backwards = bs,bptt,backwards
        self.data = self.batchify(nums)
        self.i,self.iter = 0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter<len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self): return self.n // self.bptt - 1

    def batchify(self, data):
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb*self.bs])
        data = data.reshape(self.bs, -1).T
        if self.backwards: data=data[::-1]
        return data

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].reshape([-1])


trn_lm = np.load('process/lm/train_ids.npy')
val_lm = np.load('process/lm/test_ids.npy')

trn_lm = np.concatenate(trn_lm) # 1d tensor
val_lm = np.concatenate(val_lm)

assert trn_lm.dtype == np.int64, 'Error: dtype is not int64'
assert val_lm.dtype == np.int64, 'Error: dtype is not int64'

m = LanguageModelLoader(trn_lm, 32, 512)
print(len(m))

for x, y in m:
    print(x.shape, y.shape)
    exit()
# FIXME: how to mask sequence