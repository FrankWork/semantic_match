python 2.7、java 8、jieba 0.39、
pandas 0.22.0、sklearn 0.19.1、
gensim 3.4.0、xgboost 0.71、lightgbm 2.1.1
tensorflow 1.5、pytorch 0.4.0、keras 2.1.6、


iter 50 for test
iter 1344 for train


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
rnet       21467     32     81          81      57    52    54.7/71.4  no    no  no

BirPM snli 86.9 quora 88.69
ESIM  snli 88.0 
Datt  snli 86.8

