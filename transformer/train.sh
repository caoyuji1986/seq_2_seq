export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:src/

nohup python3.6 src/main.py \
  --do_train=True \
  --do_predict=False \
  --model_dir=./out/iwslt2016 \
  --num_train_samples=190000 \
  --num_epoches=20 \
  --batch_size=32 \
  --data_dir=dat/iwslt2016/de_en/ \
  --bpe_model_file=dat/iwslt2016/de_en/model \
  --model_type=transformer \
  --model_config=./cfg/transformer.json > log.txt 2>&1 &
