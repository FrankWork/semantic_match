from pathlib import Path
import random
import fire
import re


digit_re = re.compile('\*+|\d+|\u200b|\u200c|\u200d|\u202d|\u202c|\ufeff')
space_re = re.compile('\s+', flags=re.U)

emoji_re = re.compile('\[.+?\]')

punct_zh = "！｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.·"
punct_en = "!\"#$%&\'()*+./:;<=>@[\\]^_`{|}~\\\\-" # WARNING: `-` is a special token in regex

punct_re = re.compile("[{0}{1}]".format(punct_zh, punct_en), flags=re.U)

encomma_re = re.compile(',')
enquestion_re = re.compile('\?')

def clean(text):
    text = digit_re.sub('', text)
    text = emoji_re.sub('', text)
    text = space_re.sub(' ', text)
    text = punct_re.sub('', text)
    text = encomma_re.sub('，', text)
    text = enquestion_re.sub('？', text)
    return text.strip()


def train_test_split(data, test_size=0.1):
    random.seed(123)
    random.shuffle(data) # shuffle

    n = int(len(data)*test_size)
    test_set = data[:n]
    train_set = data[n:]
    return train_set, test_set

def write_data(path, data):
    with path.open('w', encoding='utf-8') as f:
        for text in data:
            f.write(text+'\n')

def main(dir_in, dir_out, test_size=0.1):
    print(f'dir_in {dir_in} dir_out {dir_out} test_size {test_size}')
    dir_in = Path(dir_in)
    assert dir_in.exists(), f'Error: {dir_in} does not exist.'
    dir_out = Path(dir_out)
    dir_out.mkdir(exist_ok=True)

    cl_data = []
    for path in [dir_in/'atec/atec_nlp_sim_train_add.csv', dir_in/'atec/atec_nlp_sim_train.csv']:
        with path.open('r', encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                text = f'{clean(parts[1])}\t{clean(parts[2])}\t{parts[3]}'
                cl_data.append(text)
    tr_cl, te_cl = train_test_split(cl_data)

    cl_path = dir_out/'cl'
    cl_path.mkdir(exist_ok=True)
    write_data(cl_path/'train.txt', tr_cl)
    write_data(cl_path/'test.txt', te_cl)
    print('write cl data')


    lm_data = set()
    for path in [dir_in/'ccks/train.txt', dir_in/'ccks/dev.txt']:
        with path.open('r', encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                lm_data.add(clean(parts[0]))
                lm_data.add(clean(parts[1]))

    for path in [dir_in/'weibo/data.txt']:
        with path.open('r', encoding="utf-8") as f:
            for line in f:
                text = clean(line)
                if text:
                    lm_data.add(text)
    print('load lm data done')

    
    # random.seed(123)
    # random.shuffle(lm_data)

    lm_path = dir_out/'lm'
    lm_path.mkdir(exist_ok=True)
    write_data(lm_path/'train.txt', lm_data)

if __name__ == '__main__': fire.Fire(main)