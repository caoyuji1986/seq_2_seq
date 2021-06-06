#coding:utf-8
import tensorflow as tf
from tensorflow.python.eager import context
import numpy as np
import math
import six

"""
	将训练参数和计算过程分离，有利模型的可扩展性
"""
def make_mask_by_value(x):
	'''
	:param x: tensor with dtype as tf.int32
	:return: [1,1,...,1,0,0,...,0]
	'''
	zeros = tf.zeros_like(tensor=x, dtype=tf.int32)
	ones = tf.ones_like(tensor=x, dtype=tf.int32)
	x_mask = tf.where(condition=tf.equal(x=x, y=zeros), x=zeros, y=ones)
	return x_mask

class DenseOpt:
	
	def __init__(self, src_dim, dst_dim, active_fn=None, use_bias=True, name='dense'):
		
		self._src_dim = src_dim
		self._dst_dim = dst_dim
		self._active_fn = None
		initializer = tf.contrib.layers.xavier_initializer()
		self._weights = tf.get_variable(name='opt-' + name + '-weight',
		                                shape=[self._src_dim, self._dst_dim],
		                                initializer=initializer)
		self._bias = None
		if use_bias:
			self._bias = tf.get_variable(name='opt-' + name + '-bias',
		                               shape=[self._dst_dim],
		                               initializer=initializer)

	def __call__(self, x_input):
		
		#tf.assert_equal(x_input.get_shape()[-1].value, self._src_dim)
		hidden = tf.einsum('bsh,hk->bsk',x_input, self._weights)
		#hidden = tf.matmul(a=x_input, b=self._weights)
		if self._bias is not None:
			hidden = hidden +  self._bias
		if self._active_fn is not None:
			return self._active_fn(hidden)
		return hidden

def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.
  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.
  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.
  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.
  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if not context.in_eager_mode():
    if name is None:
      name = tensor.name
    if expected_rank is not None:
      assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def scaled_dot_product_attention(q, k, v, mask_q, mask_k, mask_v,
                                 attention_dropout, is_training=True,
                                 attention_future=True, dk=1):
	
	def attention_mask_before_softmax(matrix, from_mask, to_mask, attention_future=True):
		"""make sure query does not attention to key positions with value of <P>"""
		to_mask = tf.cast(x=to_mask, dtype=tf.float32)
		attention_adder = (1.0 - tf.expand_dims(input=to_mask, axis=1)) * (-2.0 ** 31 + 1.0)
		
		if attention_future == False:
			mask_matrix = tf.ones_like(matrix[0], dtype=tf.float32)
			mask_matrix = 1.0 - tf.linalg.band_part(input=mask_matrix, num_lower=-1, num_upper=0)
			mask_matrix = tf.expand_dims(input=mask_matrix, axis=0) * (-2.0 ** 31 + 1.0)
			matrix = matrix + mask_matrix  # here the first dimision will be broadcast
		
		return matrix + attention_adder  # here attention_adder will be broadcast according to axis 1
	
	def attention_mask_after_softmax(matrix, from_mask, to_mask):
		"""make sure query positions with value of <P> do not attention to any positions of key"""
		from_mask = tf.cast(x=from_mask, dtype=tf.float32)
		attention_multiplier = tf.expand_dims(input=from_mask, axis=-1)
		# B x S x S * B x S x 1
		return matrix * attention_multiplier  # here attention_multiplier will be broadcast according to axis 2
	
	# QK^T
	dot_product = tf.matmul(a=q, b=k, transpose_b=True)
	# QK^T/sqrt(dk)
	dk = tf.cast(x=dk, dtype=tf.float32)
	scale_dot_product = dot_product / tf.sqrt(dk)
	# mask & softmax
	scale_dot_product = attention_mask_before_softmax(matrix=scale_dot_product,
	                                                  from_mask=mask_q, to_mask=mask_k,
	                                                  attention_future=attention_future)
	# softmax 其实是一个比较容易膨胀的操作
	attention_weight_a = tf.nn.softmax(logits=scale_dot_product, axis=-1)
	attention_weight_a = attention_mask_after_softmax(matrix=attention_weight_a, from_mask=mask_q, to_mask=mask_k)
	
	# HERE DIFFERENT FROM PAPER
	# attention weight dropout
	attention_weight_a = tf.layers.dropout(inputs=attention_weight_a, rate=attention_dropout, training=is_training)
	# attention
	attention_score = tf.matmul(a=attention_weight_a, b=v)
	
	return attention_score

class MultiHeadAttention:
	
	def __init__(self, input_size, attention_size, attention_num, use_project=False, name='multi_head_attention'):
		
		self._attention_size = attention_size
		self._opt_dense_inst = list()
		self._attention_num = attention_num
		
		for i in range(attention_num):
			tmp_q = DenseOpt(src_dim=input_size, dst_dim=attention_size, name=name + 'q' + str(i))
			tmp_k = DenseOpt(src_dim=input_size, dst_dim=attention_size, name=name + 'k' + str(i))
			tmp_v = DenseOpt(src_dim=input_size, dst_dim=attention_size, name=name + 'v' + str(i))
			self._opt_dense_inst.append([tmp_q, tmp_k, tmp_v])
		
		self._opt_project_dense_inst = None
		if use_project:
			self._opt_project_dense_inst = DenseOpt(src_dim=attention_size*attention_num,
		                                         dst_dim=attention_size*attention_num,
		                                          name='project')
		
	def __call__(self, q, k, v, mask_k, mask_q, mask_v,
	             attention_dropout, is_training=True,
	             attention_future=True, dk=1):
		
		attention_heads = list()
		for i in range(self._attention_num):
			q_ = self._opt_dense_inst[i][0](q)
			k_ = self._opt_dense_inst[i][1](k)
			v_ = self._opt_dense_inst[i][2](v)
			attention_score = scaled_dot_product_attention(q=q_,k=k_,v=v_,
			                             mask_q=mask_q, mask_k=mask_k,mask_v=mask_v,
			                             attention_dropout=attention_dropout,is_training=is_training,
			                             attention_future=attention_future, dk=dk)
			attention_heads.append(attention_score)
		x_attention = tf.concat(values=attention_heads, axis=-1)
		
		if self._opt_project_dense_inst is not None:
			x_attention = self._opt_project_dense_inst(x_input=x_attention)
		
		return x_attention

def layer_norm(x, beta, gamma, epsilon=1e-8):
	'''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
	inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
	epsilon: A floating number. A very small number for preventing ZeroDivision Error.
	scope: Optional scope for `variable_scope`.

	Returns:
		A tensor with the same shape and data dtype as `inputs`.
	'''
	inputs = x
	inputs_shape = inputs.get_shape()
	params_shape = inputs_shape[-1:]
	
	mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
	normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
	outputs = gamma * normalized + beta
	
	return outputs


def create_position_embedding_tbl(maxlen, embeding_size, name="encoder"):
	"""
	创建论文http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf中的pos embedding
	:param maxlen: max_len for position embedding
	:param embeding_size: position embedding size
	:return: position embedding table [max_len X embedding_size]
	"""
	pos_emb = np.zeros(shape=[maxlen, embeding_size], dtype=np.float)
	wav_func = [math.sin, math.cos]
	d_model = float(embeding_size)
	for pos_ in range(maxlen):
		pos = float(pos_)
		for i_ in range(embeding_size):
			i = float(i_)
			x = pos / 10000.0 ** ((i - i_ % 2) / d_model)
			pos_emb[pos_][i_] = wav_func[i_ % 2](x)
	
	pos_embedding_lookup_tbl = tf.Variable(initial_value=pos_emb, dtype=tf.float32, name=name)
	return pos_embedding_lookup_tbl


