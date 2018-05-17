#!/bin/bash

echo $1 $2

python tokenizer.py --infile $1 --outfile test.records
python estimator.py --infile test.records --outfile $2 --model_dir model-bimpm