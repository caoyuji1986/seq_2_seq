#coding:utf-8
import tensorflow as tf
tf.enable_eager_execution()

import  numpy as np

from flag_center import FLAGS
from model import TransformerConfig, Transformer


def make_mask_by_value(x):
	'''
	:param x: tensor with dtype is tf.int32
	:return:	[1,1, ..., 1, 0,0, ... , 0]
	'''
	zeros = tf.zeros_like(tensor=x, dtype=tf.int32)
	ones = tf.ones_like(tensor=x, dtype=tf.int32)
	x_mask = tf.where(condition=tf.equal(x=x, y=zeros), x=zeros, y=ones)
	return x_mask

def beam_search(batch_size, beam_width, vocab_size, max_len, hidden_size, sos_id, eos_id, inst):
	
	# encode 端
	# batch_size x max_len
	#x_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x')
	x_placeholder = tf.convert_to_tensor(dtype=tf.int32, value=[[1,2,3,4,0,0,0,0,0,0],[5,6,7,0,0,0,0,0,0,0]])
	x_mask = make_mask_by_value(x=x_placeholder)
	# batch_size x max_len x hidden_size
	memory = inst.encode(x_input=x_placeholder, x_mask=x_mask)
	#batch_size*beam_width x max_len x hidden_size
	memorys = tf.tile(input=memory, multiples=[1, beam_width, 1])
	memorys = tf.reshape(tensor=memorys, shape=[batch_size*beam_width, max_len, hidden_size])
	# beam_width*batch_size x max_len
	memorys_mask = tf.tile(input=x_mask, multiples=[1, beam_width])
	memorys_mask = tf.reshape(tensor=memorys_mask, shape=[batch_size*beam_width, max_len])
	
	# batch_size*beam_width x 1
	y_inputs = tf.constant(value=np.ones(shape=[batch_size*beam_width, 1], dtype='int32') * sos_id, dtype=tf.int32)
	y_scores = tf.constant(value=np.zeros(shape=[batch_size*beam_width, 1], dtype='float32'), dtype=tf.float32)
	# batch_size*beam_width x 1 x vocab_size
	next_ids = tf.convert_to_tensor(value=[np.arange(0, vocab_size) for i in range(batch_size*beam_width)], dtype=tf.int32)
	next_ids = tf.reshape(tensor=next_ids, shape=[batch_size*beam_width, 1, vocab_size])
	
	def cond_fn(i, y_inputs, y_scores):
		"""
		:param i: iterator
		:param y_inputs: beam_width*batch_size x i
		:param y_scores: beam_width*batch_size x 1
		"""
		cond1 = tf.less(i, max_len)
		tmp = tf.reduce_prod(tf.cast(x=tf.equal(x=y_inputs[:, -1], y=2), dtype=tf.int32))
		cond2 = tf.equal(0, tmp)
		return tf.logical_and(x=cond1, y=cond2)
	
	def body_fn(i, y_inputs_ori, y_scores):
		"""
		:param i: iterator
		:param y_inputs: batch_size*beam_width x i
		:param y_scores: batch_size*beam_width x 1
		"""
		def padding_zeros(input, i, max_len):
		
			paddings = tf.convert_to_tensor(value=[[0 for ii in range(max_len-i-1)] for jj in range(batch_size*beam_width)], dtype=tf.int32)
			output = tf.concat(values=[input, paddings], axis=-1)
			return output
		
		#batch_size*beam_width x max_len x vocab_size
		y_inputs_ = padding_zeros(input=y_inputs_ori, i=i, max_len=max_len)
		_, scores = inst.decode(y_input=y_inputs_, y_mask=make_mask_by_value(y_inputs_),
		                        memory=memorys, memory_mask=memorys_mask)
		
		next_scores = scores[:, i, :] # batch_size*beam_width x vocab_size
		y_scores_tmp = y_scores
		# batch_size*beam_width x vocab_size
		y_scores_tmp = y_scores_tmp + tf.log(x=next_scores) # y_scores_tmp will broadcast
		y_scores_tmp = tf.reshape(tensor=y_scores_tmp, shape=[batch_size, beam_width*vocab_size])
		
		y_inputs_ori = tf.tile(input=y_inputs_ori, multiples=[1, vocab_size])
		#batch_size*beam_width  x i x vocab_size
		y_inputs_ori = tf.reshape(tensor=y_inputs_ori, shape=[beam_width*batch_size, -1, vocab_size])
		# batch_size*beam_width  x i x vocab_size, beam_width*batch_size x 1 x vocab_size -> batch_size*beam_width  x vocab_size x (i+1)
		y_inputs_ori = tf.concat(values=[y_inputs_ori, next_ids], axis=-2)
		y_inputs_ori = tf.transpose(a=y_inputs_ori, perm=[0,2,1])
		y_inputs_ori = tf.reshape(tensor=y_inputs_ori, shape=[batch_size, beam_width*vocab_size, -1])
		vals, idxs = tf.nn.top_k(input=y_scores_tmp, k=beam_width)
		y_inputs = tf.batch_gather(params=y_inputs_ori, indices=idxs)
		
		return i + 1, \
		       tf.reshape(tensor=y_inputs, shape=[batch_size*beam_width, -1]), \
		       tf.reshape(tensor=vals, shape=[batch_size*beam_width, -1])
	
	i_index_f, y_inputs, y_scores = tf.while_loop(cond=cond_fn,
	              body=body_fn,
	              loop_vars=[
		              tf.constant(0),
		              y_inputs,
		              y_scores
	              ],
	              shape_invariants=[
		              tf.TensorShape(dims=[]),
		              tf.TensorShape(dims=[batch_size*beam_width, None]),
		              tf.TensorShape(dims=[batch_size * beam_width, 1])
	              ])
	return y_inputs, y_scores
	
def main(unused_params):
	
	config = TransformerConfig.from_json_file(FLAGS.model_config)
	transformer = Transformer(config=config, mode=tf.estimator.ModeKeys.PREDICT)
	y_outputs, vals = beam_search(batch_size=2, beam_width=3, vocab_size=config.vocab_size, max_len=10, hidden_size=config.hidden_size,
	            sos_id=1, eos_id=2, inst=transformer)
	print(y_outputs)
	
	
if __name__=='''__main__''':
	
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()
	