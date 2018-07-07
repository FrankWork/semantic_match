python split_data.py data/ process_wb/
python tok2id.py process_wb/ 3000 1000

python main.py --epochs 2 --save_dir save_elm --pretrain --data_dir process/