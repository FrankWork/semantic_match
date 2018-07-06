from pathlib import Path
from collections import Counter
import re
import fire
import sys
import numpy as np


eng = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
def tokenizer(text):
    toks = []
    eng_list=[]

    for c in text:
        if len(eng_list) == 0:
            if c in eng:
                eng_list.append(c)
            else:
                toks.append(c)
        else:
            if c in eng:
                eng_list.append(c)
            else:
                toks.append(''.join(eng_list).lower())
                del eng_list[:]
                toks.append(c)
    if len(eng_list)!=0:
        toks.append(''.join(eng_list).lower())
    return toks


def get_texts(path, unsup=False):
    texts,labels = [],[]
    with path.open('r', encoding="utf-8") as f:
        if unsup:
            for line in f:
                text = line.strip()
                texts.append(text)
        else:
            for line in f:
                parts = line.strip().split('\t')
                # assert len(parts) ==2, f'len(parts)!=2 {line}'
                if len(parts) != 3:
                    continue
                label, text1, text2 = int(parts[2]), parts[0], parts[1]
                texts.append( (text1, text2) )
                labels.append(label)
    return texts, labels

def tok2id(dir_path, max_vocab=30000, min_freq=10):
    print(f'dir_path {dir_path} max_vocab {max_vocab} min_freq {min_freq}')
    p = Path(dir_path)
    assert p.exists(), f'Error: {p} does not exist.'
    lm_path = p / 'lm'
    assert lm_path.exists(), f'Error: {lm_path} does not exist.'
    cl_path = p / 'cl'
    assert cl_path.exists(), f'Error: {cl_path} does not exist.'

    # load data
    lm_tr_toks, _ = get_texts(lm_path / 'train.txt', unsup=True)

    cl_tr_toks, cl_tr_lbls = get_texts(cl_path / 'train.txt', unsup=False)
    cl_te_toks, cl_te_lbls = get_texts(cl_path / 'test.txt', unsup=False)

    print('data loading done!')
    sys.stdout.flush()

    # get vocab
    freq = Counter()
    for sent in lm_tr_toks:
        for tok in tokenizer(sent):
            freq[tok] += 1
    for sent1, sent2 in cl_te_toks+cl_tr_toks:
        for tok in tokenizer(sent1+sent2):
            freq[tok] += 1
    print(freq.most_common(25))

    vocab = [(o,c) for (o,c) in freq.most_common(max_vocab) if c>min_freq]
    with (p / 'vocab.txt').open('w') as f:
        for o, c in vocab:
            f.write(f'{o}\t{c}\n')
    print('vocab saved')
    
    itos = [o for (o,c) in vocab]
    itos.append('_bos_')
    itos.append('_eos_')
    itos.append('_unk_')
    itos.append('_delimiter_')
    itos.insert(0, '_pad_')
    np.save(p / 'itos.npy', np.array(itos))
    print(f'vocab size {len(itos)}')
    sys.stdout.flush()
    
    stoi = {v:k for k,v in enumerate(itos)}
    bid, eid, unk, delim = stoi['_bos_'], stoi['_eos_'], stoi['_unk_'], stoi['_delimiter_']

    # lm data
    
    lm_tr_ids = [ [bid] + [stoi[o] for o in tokenizer(p) if o in stoi] + [eid] for p in lm_tr_toks ]
    maxlen = max([len(s) for s in lm_tr_ids])

    lm_tr_ids = np.array(lm_tr_ids)
    np.save(lm_path/'train_ids.npy', lm_tr_ids)
    print(f'lm data saved, maxlen {maxlen}')
    sys.stdout.flush()

    # cl data
    cl_tr_ids = [ ([bid] + [stoi[o] if o in stoi else unk for o in tokenizer(p1)] + [eid], 
                   [bid] + [stoi[o] if o in stoi else unk for o in tokenizer(p2) ] + [eid]) for p1, p2 in cl_tr_toks ]
    cl_te_ids = [ ([bid] + [stoi[o] if o in stoi else unk for o in tokenizer(p1)] + [eid], 
                   [bid] + [stoi[o] if o in stoi else unk for o in tokenizer(p2)] + [eid]) for p1, p2 in cl_te_toks ]
    maxlen = max([max((len(s1), len(s2))) for (s1, s2) in cl_tr_ids] + \
                     [max((len(s1), len(s2))) for (s1, s2) in cl_te_ids])

    cl_tr_ids = np.array(cl_tr_ids)
    cl_te_ids = np.array(cl_te_ids)
    np.save(cl_path/'train_ids.npy', cl_tr_ids)
    np.save(cl_path/'test_ids.npy', cl_te_ids)
    np.save(cl_path / 'train_labels.npy', cl_tr_lbls)
    np.save(cl_path / 'test_labels.npy', cl_te_lbls)
    print(f'cl data saved, maxlen {maxlen} (without delimiter token)')
    print(f'bos {bid} eos {eid} unk {unk} delimiter {delim}')
    sys.stdout.flush()


if __name__ == '__main__': fire.Fire(tok2id)
