python 2.7、java 8、jieba 0.39、
pandas 0.22.0、sklearn 0.19.1、
gensim 3.4.0、xgboost 0.71、lightgbm 2.1.1
tensorflow 1.5、pytorch 0.4.0、keras 2.1.6、

## A 


model     epoch/step batch train_acc   test_acc p     r     f1   tune_word bn  ccks
kerasqqp   20        100                0.780               0.584      no
kerasqqp   20        100                0.780               0.584      no
kerasqqp   40        100                0.770               0.611      no
kerasqqp   26353     100    0.88        0.707               0.625      yes
kerasqqp   26870     100    0.88        0.653   0.343 0.669 0.454      yes
kerasqqp   26870     100    0.839       0.754   0.429 0.421 0.425      no
sialstm    26870     100    0.798       0.720               0.628      no             manhattan
sialstm    6870      100    0.783       0.781   0     0     0          yes   no  no   manhattan
sialstm    6870      100    0.90        0.742   0.418 0.453 0.435      yes   no  no   max, log
sialstm    6870      100    0.84        0.752   0.413 0.309 0.354      yes   no  no   basic att
sialstm    21467     32     0.90        0.751   0.438 0.473 45.4/64.7  yes   no  no   max, log
    
siacnn     26870            0.714       0.749               0.484      no    yes
siacnn     26870            0.682       0.768               0.491      no    no
esim       21078     100    0.932       0.793   0.516 0.597 0.554      no    yes yes
esim       83967     32     0.85        0.781   0.5   67.5  57.5/72.1  no    no  yes   5.47h
esim       29167     32     0.90        0.562   0.30  0.759 43.2/61.2  no    no  only  2.85h
esim       29167     32     0.92        0.802   52    60    56.3/71.5  no    no  pretrain  1.49h
esim       29167     32                 0.762   53    51    52/69      yes   no  no   0.97h
esim       21467     32     0.86        0.79    51.9  68.3  59/73.1    no    no  no   0.9h

bimpm      19389            0.96        0.794   0.529 0.554 0.541      yes   no  no   0.8714
rnet       6870      100     81         81      57    52    54.7/71.4  no    no  no   0.9154


model       snli     quora
BiMPM       86.9     88.69
ESIM        88.0 
Datt        86.8
DR-BiLSTM   88.5
DIIN        88.0
KIM         88.6
CAFE        88.5
DRCN        88.9
DMAN        88.8 

## B

model epoch batch train  test  p    r    f1      tune_w  bn  l2     ccks
rnet  20    100   89     81.9  49.7 55.4 52.4/70.7  no   no  dense   no    1.17h
rnet  20    100   86     81.5  48.8 57.6 52.9/70.9  no   no  all     no    1.14h   59.98
esim  20    100   87     80.8  47.6 61.7 53.7/71.3  no   no  all     no    0.89h   61.84
esim  20    100   86.4   81    48   63.9 54.8/71.9  no   no  all     no                  sent vec abs(-) * 
esim  20    100   85     81.7  49.4 62.9 55.3/72.3  no   no  all     no            63.12 lr decay + highway
esim  20    100   85     78.3  43.8 72.4 54.6/72    no   no  all     no    3.26h         translate
esim                                     55.0/72.1                         3.76h
esim ema        52.8/70.7
esim focal loss 53.0/71

bimpm 20    100   85     81.1  47.9 55   51.2/69.9  yes  no  all     no    1.10h   
bimpm 20    100   87     80    45.7 58.4 51.3/69.7  no   no  all     no    1.04h   
qanet 20    100   82     82    51.2 57.4 54.1/71.7  no   no  all     no    0.47h   60.89 **lr decay** no fitting
qanet 20    100   81     82.5  51.3 53.1 52.2/70.7  no   no  all     no           no ema
tflm  5+5   64           83.6  57   33   42.7/67.9  yes              yes   

# todo

- A Structured Self-Attentive Sentence Embedding
- stacking
- easy, hard

ontology, BabelNet
intent


# features

同义词 反义词 上位词 下位词
Universal Sentence Encoder
相同字符数
莱文斯坦距离(编辑距离), 欧式距离，余弦距离
SimHash
LDA/LSA
BLUE, 共同单词
pointwise mutual information PMI


# url

短文本相似度计算 https://www.zhihu.com/question/49424474
深度语义模型    https://zhuanlan.zhihu.com/p/33537217
http://www.sohu.com/a/222501203_717210
https://www.kaggle.com/c/quora-question-pairs/discussion/34325
https://www.kaggle.com/c/multinli-matched-evaluation/leaderboard

拍拍贷 https://www.ppdai.ai/mirror/goToMirrorDetail?mirrorId=1&tabindex=1
天池   https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.798f33afftdSM9&raceId=231661

# 分词

还款期 数
【 】
