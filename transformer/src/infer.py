#coding:utf-8
import tensorflow as tf
import sentencepiece as spm

from beam_search import beam_search

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer(name='max_len', default=128, help='max length')
flags.DEFINE_integer(name='beam_width', default=3, help='beam width')
flags.DEFINE_string(name='infer_file', default='', help='infer file')


def infer_beamsearch(src_tokenizer, dst_tokenizer, transformer, config):
	
	sess = tf.Session()
	saver = tf.train.Saver(var_list=tf.global_variables())
	model_file = tf.train.latest_checkpoint(FLAGS.model_dir)
	saver.restore(sess=sess, save_path=model_file)
	
	fp  = open(FLAGS.infer_file + '.dst', 'w')
	with open(FLAGS.infer_file) as fp:
		for line in fp:
			line = line.strip()
			idxs = src_tokenizer.encode_as_ids(input=line)
			idxs = idxs[:FLAGS.max_len-1]
			idxs.append(src_tokenizer.eos_id())
			for i in range(len(idxs), FLAGS.max_len):
				idxs.append(0)
			_, y_outputs, vals, x_placeholder = beam_search(batch_size=1, beam_width=FLAGS.beam_width,
			                                                vocab_size=config.vocab_size,max_len=FLAGS.max_len,
			                                                hidden_size=config.hidden_size,
			                                                sos_id=dst_tokenizer.bos_id(),
			                                                eos_id=dst_tokenizer.eos_id(),
			                                                inst=transformer)
			y_idxs, y_scores = sess.run(
				fetches=[y_outputs, vals],
				feed_dict={
					x_placeholder: [idxs]
				}
			)
			y_idxs_val = dst_tokenizer.decode_ids(input=y_idxs[0].tolist())
			fp.write(y_idxs_val + '\n')
	fp.close()


def main(unused_params):
	
	from model import TransformerConfig,RNNTransformerConfig
	from model import Transformer,RNNTransformer
	if FLAGS.model_name == 'transformer':
		config = TransformerConfig.from_json_file(FLAGS.model_config)
		transformer = Transformer(config=config, mode=tf.estimator.ModeKeys.PREDICT)
	else:
		config = RNNTransformerConfig.from_json_file(FLAGS.model_config)
		transformer = RNNTransformer(config=config, mode=tf.estimator.ModeKeys.PREDICT)
		
	bpe_model_file_src = FLAGS.bpe_model_file + '.src',
	bpe_model_file_dst = FLAGS.bpe_model_file + '.dst'
	sent_piece_src = spm.SentencePieceProcessor()
	sent_piece_src.Load(bpe_model_file_src)
	sent_piece_dst = spm.SentencePieceProcessor()
	sent_piece_dst.Load(bpe_model_file_dst)
	
	infer_beamsearch(src_tokenizer=sent_piece_src,
	                 dst_tokenizer=sent_piece_dst,
	                 transformer=transformer,
	                 config=config)
	

if __name__ == '''__main__''':

	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()


