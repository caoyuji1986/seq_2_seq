#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:src/
cd ../
nohup python3.6 src/train.py \
  --do_train=True \
  --do_predict=True \
  --model_dir=./out/wmt17 \
  --num_train_samples=190000 \
  --num_epoches=40 \
  --batch_size=128 \
  --data_dir=dat/wmt17/de_en/ \
  --bpe_model_file=dat/wmt17/model \
  --model_type=lstm \
  --model_config=./cfg/transformer.json > log.txt 2>&1 &
cd -