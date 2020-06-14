import sys
import sentencepiece as spm

file_list = [
	sys.argv[1]
]

vocab_size = 4000
model_type = 'bpe'

arg_str = "--input=%s " \
          "--model_prefix=%s " \
          "--vocab_size=%d " \
          "--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 " \
          "--model_type=%s" % (','.join(file_list), sys.argv[2], model_type)
print(arg_str)

spm.SentencePieceTrainer.train(arg_str)