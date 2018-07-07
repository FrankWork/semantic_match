

```bash
python split_data.py data/ process/
python tok2id.py process/




python split_data.py data/ process_wb/
python tok2id.py process_wb/ 3000 1000

python train.py --pretrain --lr 2.5e-4 --gpu 0 --epochs 2 --data_dir process

python train.py --pretrain --lr 2.5e-4 --gpu 0,2,3  --epochs 5 --batch_size 192  --n_embd 300 --n_layer 6 --n_head 10
```

# finetune-transformer-lm
Code and model for the paper "Improving Language Understanding by Generative Pre-Training"

