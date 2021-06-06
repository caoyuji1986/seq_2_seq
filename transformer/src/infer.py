#coding:utf-8

import tensorflow as tf
import sentencepiece as spm

from search import beam_search, greedy_search

from flag_center import FLAGS
from flag_center import flags

flags.DEFINE_integer(name='max_len', default=128, help='max length')
flags.DEFINE_integer(name='beam_width', default=3, help='beam width')
flags.DEFINE_string(name='infer_file', default='', help='infer file')
flags.DEFINE_string(name='methord', default='', help='使用的解码方法')


def infer_search(src_tokenizer, dst_tokenizer, transformer, config, methord='beam_search'):
	

	
	if methord == 'beam_search':
		_, y_outputs, _, x_placeholder = beam_search(batch_size=1, beam_width=FLAGS.beam_width,
		                                                vocab_size=config.vocab_size, max_len=FLAGS.max_len,
		                                                hidden_size=config.hidden_size,
		                                                sos_id=dst_tokenizer.bos_id(),
		                                                eos_id=dst_tokenizer.eos_id(),
		                                                inst=transformer)
	elif methord == 'greedy_search':
		_, y_outputs, x_placeholder = greedy_search(batch_size=1,
		                                                  max_len=FLAGS.max_len,
		                                                  sos_id=dst_tokenizer.bos_id(),
		                                                  eos_id=dst_tokenizer.eos_id(),
		                                                  inst=transformer)
	else:
		raise NotImplementedError('尚未支持')

	sess = tf.Session()
	saver = tf.train.Saver()
	model_file = tf.train.latest_checkpoint(FLAGS.model_dir)
	saver.restore(sess=sess, save_path=model_file)

	fpw  = open(file=FLAGS.infer_file + '.dst', mode='w', encoding='utf-8')
	with open(file=FLAGS.infer_file, mode='r', encoding='utf-8') as fp:
		for line in fp:
			line = line.strip()
			idxs = src_tokenizer.encode_as_ids(input=line)
			idxs = idxs[:FLAGS.max_len-1]
			idxs.append(src_tokenizer.eos_id())
			for i in range(len(idxs), FLAGS.max_len):
				idxs.append(0)
			y_idxs, = sess.run(
				fetches=[y_outputs],
				feed_dict={
					x_placeholder: [idxs]
				}
			)
			y_idxs_val = dst_tokenizer.decode_ids(ids=y_idxs[0].tolist())
			fpw.write(y_idxs_val + '\n')
	fpw.close()


def main(unused_params):
	
	from model import TransformerConfig,RNNTransformerConfig
	from model import Transformer,RNNTransformer
	if FLAGS.model_type == 'transformer':
		config = TransformerConfig.from_json_file(FLAGS.model_config)
		transformer = Transformer(config=config, mode=tf.estimator.ModeKeys.PREDICT)
	else:
		config = RNNTransformerConfig.from_json_file(FLAGS.model_config)
		transformer = RNNTransformer(config=config, mode=tf.estimator.ModeKeys.PREDICT)
		
	bpe_model_file_src = FLAGS.bpe_model_file + '.src'
	bpe_model_file_dst = FLAGS.bpe_model_file + '.dst'
	sent_piece_src = spm.SentencePieceProcessor()
	sent_piece_src.Load(bpe_model_file_src)
	sent_piece_dst = spm.SentencePieceProcessor()
	sent_piece_dst.Load(bpe_model_file_dst)


	infer_search(src_tokenizer=sent_piece_src, dst_tokenizer=sent_piece_dst,
	             transformer=transformer, config=config, methord=FLAGS.methord)
	

if __name__ == '''__main__''':

	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()


