#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:src/
cd ../
nohup python3.6 src/train.py \
  --do_train=True \
  --do_predict=True \
  --model_dir=./out/iwslt2016 \
  --num_train_samples=190000 \
  --num_epoches=40 \
  --batch_size=128 \
  --data_dir=dat/iwslt2016/de_en/ \
  --bpe_model_file=dat/iwslt2016/de_en/model \
  --model_type=lstm \
  --model_config=./cfg/rnn.json > log.txt 2>&1 &
cd -