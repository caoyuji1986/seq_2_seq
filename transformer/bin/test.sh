#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:src/

cd ../
nohup python3.6 src/infer.py \
  --model_dir=./out/iwslt2016 \
  --data_dir=dat/iwslt2016/de_en/ \
  --bpe_model_file=dat/iwslt2016/de_en/model \
  --model_type=lstm\
  --model_config=./cfg/rnn.json > log.txt 2>&1 &
cd -

