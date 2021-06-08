import sys
import sentencepiece as spm

file_list = [
	"./dat/wmt17/corpus.tc.de"
    ,"./dat/wmt17/corpus.tc.en"
    ,"./dat/wmt17/newstest2016.tc.de"
    ,"./dat/wmt17/newstest2016.tc.en"
]

vocab_size = 32000
model_type = 'bpe'

arg_str = "--input=%s " \
          "--model_prefix=%s " \
          "--vocab_size=%d " \
          "--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 " \
          "--model_type=%s" % (','.join(file_list), model_type, vocab_size, model_type)
print(arg_str)

spm.SentencePieceTrainer.train(arg_str)