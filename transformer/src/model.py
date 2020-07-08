#coding:utf-8
from operation import *
import tensorflow as tf
import six
import copy
import json

class BaseTransformer():
	
	def __init__(self):
		pass
	
	@staticmethod
	def label_smoothing(inputs, epsilon=0.1):
		inputs = tf.cast(x=inputs, dtype=tf.float32)
		num_channels = inputs.get_shape().as_list()[-1]
		return (1.0 - epsilon) * inputs + epsilon / num_channels


class TransformerConfig(object):
	"""Configuration for `TransformerModel`."""
	
	def __init__(self,
	             vocab_size=37000,
	             hidden_size=512,
	             num_hidden_layers=6,
	             attention_size=64,
	             position_wise_feed_forward_size=2048,
	             embedding_dropout_prob=0.1,
	             sub_layer_dropout_prob=0.1,
	             max_position_embeddings=512,
	             init_std=0.02):
		"""
			vocab_size: bpe词表的大小, nmt 一定要用bpe
			hidden_size: 隐层的宽度 d_model
			num_hidden_layers: 编解码的层数 h
			attention_size: multi-head attention attention size d_k or d_v
			position_wise_feed_forward_size: feadforward 层的宽度 dff
			embedding_dropout_prob: Embedding dropout
			sub_layer_dropout_prob: Sublayer dropout
			max_position_embeddings: 最大的position 的长度
			init_std: 变量初始化参数
		"""
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.attention_size = attention_size
		self.position_wise_feed_forward_size = position_wise_feed_forward_size
		self.embedding_dropout_prob = embedding_dropout_prob
		self.sub_layer_dropout_prob = sub_layer_dropout_prob
		self.max_position_embeddings = max_position_embeddings
		self.init_std = init_std
	
	@classmethod
	def from_dict(cls, json_object):
		"""Constructs a `TransformerConfig` from a Python dictionary of parameters."""
		config = TransformerConfig(vocab_size=None)
		for (key, value) in six.iteritems(json_object):
			config.__dict__[key] = value
		return config
	
	@classmethod
	def from_json_file(cls, json_file):
		"""Constructs a `TransformerConfig` from a json file of parameters."""
		with tf.gfile.GFile(json_file, "r") as reader:
			text = reader.read()
		return cls.from_dict(json.loads(text))
	
	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output
	
	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Transformer(BaseTransformer):
	
	def __init__(self, config, mode):
		"""
		:type config: TransformerConfig
		:type mode: tf.estimator.ModeKeys
		"""
		self._config = config
		"""
		[NOTICE] 编码和解码使用同一张embedding表，但是使用不同的pos embedding表
		"""
		with tf.variable_scope(name_or_scope='word_embedding', reuse=tf.AUTO_REUSE):
			"""
			[NOTICE1] embedding table 必须使用xavier_initializer
			[NOTICE2] token0 必须初始化成0向量
			"""
			embedding_lookup_tbl_encoder = tf.get_variable(name='embedding_lookup_tbl_encoder',
																								 shape=[self._config.vocab_size, self._config.hidden_size],
																								 dtype=tf.float32,
																								 initializer=tf.contrib.layers.xavier_initializer())
			zero_emb_encoder = tf.zeros(name='embedding_lookup_tbl_encoder_zero',
			                            shape=[1, self._config.hidden_size],
			                            dtype=tf.float32)
			self._embedding_lookup_tbl_encoder = \
				tf.concat(values=[zero_emb_encoder, embedding_lookup_tbl_encoder[1:]], axis=0)

			embedding_lookup_tbl_decoder = tf.get_variable(name='embedding_lookup_tbl_decoder',
																								 shape=[self._config.vocab_size, self._config.hidden_size],
																								 dtype=tf.float32,
																								 initializer=tf.contrib.layers.xavier_initializer())
			zero_emb_decoder = tf.zeros(name='embedding_lookup_tbl_decoder_zero',
			                            shape=[1, self._config.hidden_size],
			                            dtype=tf.float32)
			self._embedding_lookup_tbl_decoder = \
				tf.concat(values=[zero_emb_decoder, embedding_lookup_tbl_decoder[1:]], axis=0)

		with tf.variable_scope(name_or_scope='position_embedding', reuse=tf.AUTO_REUSE):
			self._pos_embedding_lookup_tbl_encoder = create_position_embedding_tbl(self._config.max_position_embeddings,
																															 self._config.hidden_size, "encoder")
			self._pos_embedding_lookup_tbl_decoder = create_position_embedding_tbl(self._config.max_position_embeddings,
																															 self._config.hidden_size, "decoder")

		
		with tf.variable_scope(name_or_scope='encoder'):
			self._encoder_opt = list()
			for layer_index in range(self._config.num_hidden_layers):
				att = MultiHeadAttention(input_size=self._config.hidden_size,
				                         attention_size=self._config.attention_size,
				                         attention_num=int(self._config.hidden_size / self._config.attention_size),
				                         name='mha' + str(layer_index))
				beta1 = tf.get_variable(name="layer_norm_1_beta" + str(layer_index),
				                        shape=self._config.hidden_size,
				                        initializer=tf.zeros_initializer())
				gamma1 = tf.get_variable(name="layer_norm_1_gamma" + str(layer_index),
				                         shape=self._config.hidden_size,
				                         initializer=tf.ones_initializer())
				layer_norm_1 =(beta1, gamma1)
				feed_forward_inner = DenseOpt(src_dim=self._config.hidden_size,
				                              dst_dim=self._config.position_wise_feed_forward_size,
				                              active_fn=tf.nn.relu,
				                              name='feed_forward_inner' + str(layer_index))
				feed_forward_outer = DenseOpt(src_dim=self._config.position_wise_feed_forward_size,
				                              dst_dim=self._config.hidden_size,
				                              name='feed_forward_outer' + str(layer_index))
				feed_forward = (feed_forward_inner, feed_forward_outer)
				beta2 = tf.get_variable("layer_norm_2_beta" + str(layer_index), self._config.hidden_size,
				                       initializer=tf.zeros_initializer())
				gamma2 = tf.get_variable("layer_norm_2_gamma" + str(layer_index), self._config.hidden_size,
				                        initializer=tf.ones_initializer())
				layer_norm_2 = (beta2, gamma2)
				self._encoder_opt.append({
					'att': att,
					'layer_norm_1': layer_norm_1,
					'feed_forward': feed_forward,
					'layer_norm_2': layer_norm_2
				})
				
		with tf.variable_scope(name_or_scope='decoder'):
			self._decoder_opt = list()
			for layer_index in range(self._config.num_hidden_layers):
				att1 = MultiHeadAttention(input_size=self._config.hidden_size,
				                         attention_size=self._config.attention_size,
				                         attention_num=int(self._config.hidden_size / self._config.attention_size),
				                         name='mha1_' + str(layer_index))
				beta1 = tf.get_variable("layer_norm_1_beta" + str(layer_index), self._config.hidden_size,
				                       initializer=tf.zeros_initializer())
				gamma1 = tf.get_variable("layer_norm_1_gamma" + str(layer_index), self._config.hidden_size,
				                        initializer=tf.ones_initializer())
				layer_norm_1 =(beta1, gamma1)
				att2 = MultiHeadAttention(input_size=self._config.hidden_size,
				                          attention_size=self._config.attention_size,
				                          attention_num=int(self._config.hidden_size / self._config.attention_size),
				                          name='mha2_' + str(layer_index))
				beta2 = tf.get_variable("layer_norm_2_beta" + str(layer_index), self._config.hidden_size,
				                        initializer=tf.zeros_initializer())
				gamma2 = tf.get_variable("layer_norm_2_gamma" + str(layer_index), self._config.hidden_size,
				                         initializer=tf.ones_initializer())
				layer_norm_2 = (beta2, gamma2)
				feed_forward_inner = DenseOpt(src_dim=self._config.hidden_size, dst_dim=self._config.position_wise_feed_forward_size,
				                        active_fn=tf.nn.relu, name='feed_forward_inner' + str(layer_index))
				feed_forward_outer = DenseOpt(src_dim=self._config.position_wise_feed_forward_size, dst_dim=self._config.hidden_size,
				                        name='feed_forward_outer' + str(layer_index))
				feed_forward = (feed_forward_inner, feed_forward_outer)
				beta3 = tf.get_variable("layer_norm_3_beta" + str(layer_index), self._config.hidden_size,
				                       initializer=tf.zeros_initializer())
				gamma3 = tf.get_variable("layer_norm_3_gamma" + str(layer_index), self._config.hidden_size,
				                        initializer=tf.ones_initializer())
				layer_norm_3 = (beta3, gamma3)
				self._decoder_opt.append({
					'att1': att1,
					'layer_norm_1': layer_norm_1,
					'att2': att2,
					'layer_norm_2': layer_norm_2,
					'feed_forward': feed_forward,
					'layer_norm_3': layer_norm_3
				})
		self._mode = mode
	
	def encode(self, x_input, x_mask):
		
		x_id_emb = tf.nn.embedding_lookup(params=self._embedding_lookup_tbl_encoder, ids=x_input)
		x_id_emb *= self._config.hidden_size ** 0.5  # IMPORTANT !!!!!
		seq_len = get_shape_list(x_input)[1]
		x_position_emb = tf.slice(input_=self._pos_embedding_lookup_tbl_encoder, begin=[0, 0], size=[seq_len, -1])
		x_position_emb = x_position_emb * tf.cast(x=tf.expand_dims(input=x_mask, axis=-1), dtype=tf.float32)
		x = x_id_emb + x_position_emb  # B x S x H 请注意在生成特征的时候不要超过position embedding的最大长度
		# model regularization
		x = tf.layers.dropout(inputs=x, rate=self._config.embedding_dropout_prob,
		                      training=self._mode == tf.estimator.ModeKeys.TRAIN)
		
		for layer_index in range(self._config.num_hidden_layers):
			x_sub_layer = self._encoder_opt[layer_index]['att'](q=x, k=x, v=x,
			                                      mask_k=x_mask,mask_q=x_mask, mask_v=x_mask,
			                                      attention_dropout=self._config.sub_layer_dropout_prob,
			                                      is_training=self._mode == tf.estimator.ModeKeys.TRAIN,
			                                      dk=self._config.attention_size
			                                      )
			layer_norm_1 = self._encoder_opt[layer_index]['layer_norm_1']
			x_add_norm_1 = layer_norm(x=x + x_sub_layer, beta=layer_norm_1[0], gamma=layer_norm_1[1])
			
			feed_forward_inner = self._encoder_opt[layer_index]['feed_forward'][0]
			feed_forward_outer = self._encoder_opt[layer_index]['feed_forward'][1]
			x_ffn_inner = feed_forward_inner(x_input=x_add_norm_1)
			x_ffn_outer = feed_forward_outer(x_input=x_ffn_inner)
			
			layer_norm_2 = self._encoder_opt[layer_index]['layer_norm_2']
			x_add_norm_2 = layer_norm(x=x_add_norm_1 + x_ffn_outer, beta=layer_norm_2[0], gamma=layer_norm_2[1])
			x = x_add_norm_2

		return x, x_mask
		
	def decode(self, y_input, y_mask, memory, memory_mask):

		y_id_emb = tf.nn.embedding_lookup(params=self._embedding_lookup_tbl_decoder, ids=y_input)
		y_id_emb *= self._config.hidden_size ** 0.5  # IMPORTANT !!!!!
		seq_len = get_shape_list(y_input)[1]
		y_position_emb = tf.slice(input_=self._pos_embedding_lookup_tbl_encoder, begin=[0, 0], size=[seq_len, -1])
		y_position_emb = y_position_emb * tf.cast(x=tf.expand_dims(input=y_mask, axis=-1), dtype=tf.float32)
		y = y_id_emb + y_position_emb  # B x S x H 请注意在生成特征的时候不要超过position embedding的最大长度
		# model regularization
		y = tf.layers.dropout(inputs=y, rate=self._config.embedding_dropout_prob,
		                      training=self._mode == tf.estimator.ModeKeys.TRAIN)
		
		for layer_index in range(self._config.num_hidden_layers):
			y_sub_layer_1 = self._decoder_opt[layer_index]['att1'](q=y, k=y, v=y,
			                                                    mask_k=y_mask, mask_q=y_mask, mask_v=y_mask,
			                                                    attention_dropout=self._config.sub_layer_dropout_prob,
			                                                    is_training=self._mode == tf.estimator.ModeKeys.TRAIN,
			                                                    dk=self._config.attention_size, attention_future=False
			                                                    )
			layer_norm_1 = self._decoder_opt[layer_index]['layer_norm_1']
			y_add_norm_1 = layer_norm(x=y + y_sub_layer_1, beta=layer_norm_1[0], gamma=layer_norm_1[1])
			
			y_sub_layer_2 = self._decoder_opt[layer_index]['att2'](q=y_add_norm_1, k=memory, v=memory,
			                                                    mask_k=memory_mask, mask_q=y_mask, mask_v=memory_mask,
			                                                    attention_dropout=self._config.sub_layer_dropout_prob,
			                                                    is_training=self._mode == tf.estimator.ModeKeys.TRAIN,
			                                                    dk=self._config.attention_size, attention_future=True
			                                                    )
			layer_norm_2 = self._decoder_opt[layer_index]['layer_norm_2']
			y_add_norm_2 = layer_norm(x=y_add_norm_1 + y_sub_layer_2, beta=layer_norm_2[0], gamma=layer_norm_2[1])
			
			feed_forward_inner = self._decoder_opt[layer_index]['feed_forward'][0]
			feed_forward_outer = self._decoder_opt[layer_index]['feed_forward'][1]
			y_ffn_inner = feed_forward_inner(x_input=y_add_norm_2)
			y_ffn_outer = feed_forward_outer(x_input=y_ffn_inner)
			
			layer_norm_3 = self._decoder_opt[layer_index]['layer_norm_3']
			y_add_norm_3 = layer_norm(x=y_add_norm_2 + y_ffn_outer, beta=layer_norm_3[0], gamma=layer_norm_3[1])
			y = y_add_norm_3
			
		weights = tf.transpose(a=self._embedding_lookup_tbl_decoder)
		logits = tf.einsum('ntd,dk->ntk', y, weights)
		scores = tf.nn.softmax(logits=logits, axis=-1)
		
		return logits, scores
	
	def create_model(self, x_input, y_input):
		
		x_mask = make_mask_by_value(x=x_input)
		y_mask = make_mask_by_value(x=y_input)
		
		memory = self.encode(x_input=x_input, x_mask=x_mask)
		logits, scores = self.decode(y_input=y_input, y_mask=y_mask, memory=memory, memory_mask=x_mask)
		
		return logits, scores
	
	def calculate_loss(self, logits, y_labels):
		
		y_label_mask = make_mask_by_value(x=y_labels)
		log_probs = tf.nn.log_softmax(logits=logits, axis=-1)
		one_hot_labels = tf.one_hot(indices=y_labels, depth=self._config.vocab_size, dtype=tf.float32)
		smoothed_one_hot_labels = super().label_smoothing(inputs=one_hot_labels)
		per_sample_loss = -tf.reduce_sum(input_tensor=(smoothed_one_hot_labels * log_probs), axis=-1)
		per_sample_loss = per_sample_loss * tf.cast(x=y_label_mask, dtype=tf.float32)
		loss = tf.reduce_sum(input_tensor=per_sample_loss) / tf.reduce_sum(tf.cast(x=y_label_mask, dtype=tf.float32))
		
		return loss
		

class RNNTransformerConfig:
	
	def __init__(self,
	             vocab_size=4000,
	             hidden_size=512,
	             cell_dropout_prob=0.0,
	             attention_dropout_prob=0.1,
				 embedding_dropout_prob=0.1,
	             use_attention=True,
	             cell_type='lstm',
				 num_hidden_layers=4,
				 use_residual=True,
				 reverse_encode_input=True):
		"""
			vocab_size: bpe词表的大小, nmt 一定要用bpe
			hidden_size: 隐层的宽度 d_model
			cell_dropout_prob: cell dropout prob
			attention_dropout_prob: attention dropout prob
			use_attention: 是不是使用attention
			cell_type: 使用什么样的RNN单元
		"""
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.cell_dropout_prob = cell_dropout_prob
		self.attention_dropout_prob = attention_dropout_prob
		self.embedding_dropout_prob = embedding_dropout_prob
		self.use_attention = use_attention
		self.cell_type = cell_type
		self.num_hidden_layers = num_hidden_layers
		self.use_residual = use_residual
		self.reverse_encode_input = reverse_encode_input
		
	@classmethod
	def from_dict(cls, json_object):
		"""Constructs a `TransformerConfig` from a Python dictionary of parameters."""
		config = RNNTransformerConfig(vocab_size=None)
		for (key, value) in six.iteritems(json_object):
			config.__dict__[key] = value
		return config
	
	@classmethod
	def from_json_file(cls, json_file):
		"""Constructs a `TransformerConfig` from a json file of parameters."""
		with tf.gfile.GFile(json_file, "r") as reader:
			text = reader.read()
		return cls.from_dict(json.loads(text))
	
	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output
	
	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class RNNTransformer(BaseTransformer):
	
	def __init__(self, config, mode):
		
		self._config = config
		self._mode = mode
		initializer = tf.random_uniform_initializer(minval=-0.08, maxval=0.08)
		cell_keep_prob = 1.0 - self._config.cell_dropout_prob if self._mode == tf.estimator.ModeKeys.TRAIN else 1.0
		
		if self._config.cell_type == 'lstm':
			cell = tf.nn.rnn_cell.BasicLSTMCell
		elif self._config.cell_type == 'gru':
			cell = tf.nn.rnn_cell.GRUCell
		else:
			cell = tf.nn.rnn_cell.RNNCell

		with tf.variable_scope(name_or_scope='encoder', initializer=initializer):
			"""
				[NOTICE2] token0 必须初始化成0向量
			"""
			embedding_lookup_tbl_encoder = tf.get_variable(name='embedding_lookup_tbl_encoder',
														   shape=[self._config.vocab_size, self._config.hidden_size],
														   dtype=tf.float32
														   )
			zero_emb_encoder = tf.zeros(shape=[1, self._config.hidden_size], dtype=tf.float32)
			self._embedding_lookup_tbl_encoder = tf.concat(values=[zero_emb_encoder, embedding_lookup_tbl_encoder[1:]],
														   axis=0)
			self._embedding_bias_encoder = tf.get_variable(name='encoder_emb_bias',
														   shape=[self._config.hidden_size],
														   dtype=tf.float32
														   )

			cells = [tf.nn.rnn_cell.DropoutWrapper(
								cell=cell(num_units=self._config.hidden_size, name='encoder_cell_%d' %(i)),
								output_keep_prob=cell_keep_prob)
						for i in range(self._config.num_hidden_layers)]
			if self._config.use_residual:
				cells = [tf.nn.rnn_cell.ResidualWrapper(cell=cell) for cell in cells]
			self._encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells)

		with tf.variable_scope(name_or_scope='decoder', initializer=initializer):
			"""
				[NOTICE2] token0 必须初始化成0向量
			"""
			embedding_lookup_tbl_decoder = tf.get_variable(name='embedding_lookup_tbl_decoder',
														   shape=[self._config.vocab_size, self._config.hidden_size],
														   dtype=tf.float32,
														   initializer=initializer)
			zero_emb_decoder = tf.zeros(shape=[1, self._config.hidden_size], dtype=tf.float32)
			self._embedding_lookup_tbl_decoder = tf.concat(values=[zero_emb_decoder, embedding_lookup_tbl_decoder[1:]],
														   axis=0)
			self._embedding_bias_decoder = tf.get_variable(name='decoder_emb_bias',
														   shape=[self._config.hidden_size],
														   dtype=tf.float32,
														   initializer=initializer)

			cells = [tf.nn.rnn_cell.DropoutWrapper(
								cell=cell(num_units=self._config.hidden_size, name='decoder_cell_%d' % (i)),
								output_keep_prob=cell_keep_prob)
				for i in range(self._config.num_hidden_layers)]
			if self._config.use_residual:
				cells = [tf.nn.rnn_cell.ResidualWrapper(cell=cell) for cell in cells]
			self._decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
		
		self._project = DenseOpt(src_dim=self._config.hidden_size, dst_dim=self._config.vocab_size, use_bias=True)

	def encode(self, x_input, x_mask):
		if self._config.reverse_encode_input:
			# 因为论文中讲如果逆转编码端的输入，BLUE值会有大幅度提升
			x_length = tf.reduce_sum(input_tensor=x_mask, axis=-1)
			x_input = tf.reverse_sequence(input=x_input, seq_lengths=x_length, seq_dim=1)
		x_input = tf.nn.embedding_lookup(params=self._embedding_lookup_tbl_encoder, ids=x_input)
		x_input = tf.nn.bias_add(value=x_input, bias=self._embedding_bias_encoder)
		x_input = tf.layers.dropout(inputs=x_input, rate=self._config.embedding_dropout_prob,
									training=tf.estimator.ModeKeys.TRAIN==self._mode)

		sequence_len = tf.reduce_sum(input_tensor=x_mask, axis=-1)
		batch_size = get_shape_list(tensor=x_input)[0]
		encoder_initial_state = self._encoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
		outputs, last_states = tf.nn.dynamic_rnn(cell=self._encoder_cell, inputs=x_input,
		                                         initial_state=encoder_initial_state,
		                                         sequence_length=sequence_len)
		return outputs, last_states
	
	def decode(self, y_input, y_mask, memory, memory_mask):

		outputs = memory[0]
		last_state = memory[1]
		y_input = tf.nn.embedding_lookup(params=self._embedding_lookup_tbl_decoder, ids=y_input)
		y_input = tf.nn.bias_add(value=y_input, bias=self._embedding_bias_decoder)
		y_input = tf.layers.dropout(inputs=y_input, rate=self._config.embedding_dropout_prob,
									training=tf.estimator.ModeKeys.TRAIN == self._mode)

		sequence_len = tf.reduce_sum(input_tensor=y_mask, axis=-1)
		outputs_decoder, last_states_decoder = tf.nn.dynamic_rnn(cell=self._decoder_cell, inputs=y_input,
		                                         initial_state=last_state, sequence_length=sequence_len)

		if self._config.use_attention:
			outputs_ = scaled_dot_product_attention(q=outputs_decoder, k=outputs, v=outputs,
		                             mask_q=y_mask, mask_k=memory_mask, mask_v=memory_mask,
		                             attention_dropout=self._config.attention_dropout_prob)
		else:
			outputs_ = outputs_decoder
		
		#logits = self._project(x_input=outputs_)
		weights = tf.transpose(a=self._embedding_lookup_tbl_decoder)
		logits = tf.einsum('ntd,dk->ntk', outputs_, weights)
		scores = tf.nn.softmax(logits=logits, axis=-1)
		
		return logits, scores

	def create_model(self, x_input, y_input):

		x_mask = make_mask_by_value(x=x_input)
		memory = self.encode(x_input=x_input, x_mask=x_mask)
		y_mask = make_mask_by_value(x=y_input)
		logtis, scores = self.decode(y_input=y_input, y_mask=y_mask, memory=memory, memory_mask=x_mask)
		
		return logtis, scores
	
	def calculate_loss(self, logits, y_labels):
	
		y_label_mask = make_mask_by_value(x=y_labels)
		log_probs = tf.nn.log_softmax(logits=logits, axis=-1)
		one_hot_labels = tf.one_hot(indices=y_labels, depth=self._config.vocab_size, dtype=tf.float32)
		smoothed_one_hot_labels = super().label_smoothing(inputs=one_hot_labels)
		per_sample_loss = -tf.reduce_sum(input_tensor=(smoothed_one_hot_labels * log_probs), axis=-1)
		per_sample_loss = per_sample_loss * tf.cast(x=y_label_mask, dtype=tf.float32)
		loss = tf.reduce_sum(input_tensor=per_sample_loss) / tf.reduce_sum(tf.cast(x=y_label_mask, dtype=tf.float32))
		
		return loss


class ConvTransformerConfig:

	def __init__(self,
				 vocab_size=37000,
				 hidden_size=512,
				 num_hidden_layers=6,
				 attention_size=64,
				 embedding_dropout_prob=0.1,
				 sub_layer_dropout_prob=0.1,
				 max_position_embeddings=512,
				 init_std=0.02,
				 kernel_list=[3,3,3,4]
				 ):

		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.attention_size = attention_size
		self.embedding_dropout_prob = embedding_dropout_prob
		self.sub_layer_dropout_prob = sub_layer_dropout_prob
		self.max_position_embeddings = max_position_embeddings
		self.init_std = init_std
		self.kernel_list = kernel_list

	@classmethod
	def from_dict(cls, json_object):
		"""Constructs a `TransformerConfig` from a Python dictionary of parameters."""
		config = ConvTransformerConfig(vocab_size=None)
		for (key, value) in six.iteritems(json_object):
			config.__dict__[key] = value
		return config

	@classmethod
	def from_json_file(cls, json_file):
		"""Constructs a `TransformerConfig` from a json file of parameters."""
		with tf.gfile.GFile(json_file, "r") as reader:
			text = reader.read()
		return cls.from_dict(json.loads(text))

	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class ConvTransformer(BaseTransformer):

	def __init__(self, config, mode):

		self._config = config
		self._mode = mode

		with tf.variable_scope(name_or_scope='encoder'):
			self._pos_embedding_lookup_tbl_encoder = tf.get_variable(
				name="pos",
				shape=[self._config.max_position_embeddings, self._config.hidden_size],
				initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
			self._embedding_lookup_tbl_encoder = tf.get_variable(
				name="id",
				shape=[self._config.vocab_size, self._config.hidden_size],
				initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1)
			)
			self._kernal_list_encoder = list()
			for i in range(len(self._config.kernel_list)):
				kernel1 = tf.get_variable(name='kernel1_' + str(i),
							shape=[self._config.kernel_list[i], 1, self._config.hidden_size, self._config.hidden_size],
							initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

				kernel2 = tf.get_variable(name='kernel2_' + str(i),
							shape=[self._config.kernel_list[i], 1, self._config.hidden_size, self._config.hidden_size],
							initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
				self._kernal_list_encoder.append([kernel1, kernel2])


		with tf.variable_scope(name_or_scope='decoder'):
			self._pos_embedding_lookup_tbl_decoder = tf.get_variable(
				name="pos",
				shape=[self._config.max_position_embeddings, self._config.hidden_size],
				initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
			self._embedding_lookup_tbl_decoder = tf.get_variable(
				name="id",
				shape=[self._config.vocab_size, self._config.hidden_size],
				initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1)
			)
			self._kernal_list_decoder = list()
			self._project_list_decoder = list()
			for i in range(len(self._config.kernel_list)):
				kernel1 = tf.get_variable(name='kernel1_' + str(i),
										 shape=[self._config.kernel_list[i], 1, self._config.hidden_size,
												self._config.hidden_size],
										 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
				kernel2 = tf.get_variable(name='kernel2_' + str(i),
										 shape=[self._config.kernel_list[i], 1, self._config.hidden_size,
												self._config.hidden_size],
										 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
				self._kernal_list_decoder.append([kernel1, kernel2])
				dense_opt = DenseOpt(src_dim=self._config.hidden_size,
									 dst_dim=self._config.hidden_size, name='decode_dense_' + str(i))
				self._project_list_decoder.append(dense_opt)

	def __reverse(self, x):
		'''
		[TODO] 需要验证正确性
		'''

		x_tensor = x
		lengths = tf.reduce_sum(input_tensor=tf.sign(x=x_tensor), axis=-1)
		# [[1,2,3,4,PAD,PAD,PAD],[2,3,PAD,PAD,PAD,PAD,PAD]]   [4,2]
		x_tensor = tf.reverse_sequence(x_tensor, lengths, batch_dim=0, seq_dim=1)

		# --> [[4,3,2,1,PAD,PAD,PAD],[3,2,PAD,PAD,PAD,PAD,PAD]] --> [[PAD,PAD,PAD,1,2,3,4],[PAD,PAD,PAD,PAD,PAD,2,3]]
		x_tensor = tf.reverse(x_tensor, [1])

		return x_tensor


	def __embedding(self, x_input, x_mask, x_emb_tbl, pos_emb_tbl):

		# word embedding
		x_emb = tf.nn.embedding_lookup(params=x_emb_tbl, ids=x_input)
		# pos embedding
		batch_size, max_len = get_shape_list(tensor=x_input)[:2]
		start_pos = max_len - tf.reduce_sum(input_tensor=x_mask, axis=-1)
		r = tf.reshape(tensor=tf.tile(input=tf.range(start=0, limit=max_len), multiples=[batch_size]),
					   shape=[batch_size, -1])
		start_pos = r - tf.expand_dims(input=start_pos, axis=-1) + 1
		zero_pos = tf.zeros_like(tensor=start_pos)
		start_pos = tf.where(condition=tf.less(x=start_pos, y=zero_pos), x=zero_pos, y=start_pos)
		pos_emb = tf.nn.embedding_lookup(params=pos_emb_tbl, ids=start_pos)

		return x_emb + pos_emb

	def encode(self, x_input, x_mask):
		'''
		:param x_input: [[x1,x2,x3,eos_id,P,P,P],[x1,x2,x3,x4,eos_id,P,P]]
		:param x_mask: [[1,1,1,1,P,P,P],[1,1,1,1,1,P,P]]
		:return:
		'''
		# [[P, P, P, x1, x2, x3, eos_id], [P, P, x1, x2, x3, x4, eos_id]]
		x_input = self.__reverse(x=x_input)
		# [[0,0,0,1,1,1,1],[0,0,1,1,1,1,1]]
		x_mask = self.__reverse(x=x_mask)

		# batch_size x seq_len x hidden_size
		x_emb = self.__embedding(x_input=x_input,
								 x_mask=x_mask,
								 x_emb_tbl=self._embedding_lookup_tbl_encoder,
								 pos_emb_tbl=self._pos_embedding_lookup_tbl_encoder)
		# batch_size x seq_len x 1 x hidden_size
		x_hidden = tf.expand_dims(input=x_emb, axis=-2)

		for i in range(len(self._config.kernel_list)):
			kernel_size = self._config.kernel_list[i]
			x_hidden_pre = x_hidden
			x_hidden = tf.pad(tensor=x_hidden, paddings=[[0, 0], [kernel_size - 1, 0], [0,0], [0,0]], constant_values=0)
			# batch_size x seq_len x 1 x hidden_size
			x1 = tf.nn.conv2d(input=x_hidden, filter=self._kernal_list_encoder[i][0], padding='VALID', strides=[1,1,1,1])
			x2 = tf.nn.conv2d(input=x_hidden, filter=self._kernal_list_encoder[i][1], padding='VALID', strides=[1,1,1,1])

			input_gate = tf.sigmoid(x=x1)
			x_hidden = x2 * input_gate
			x_hidden = x_hidden * x_mask
			# resnet
			x_hidden = (x_hidden + x_hidden_pre) * tf.sqrt(0.5)

		return x_hidden, x_emb

	def decode(self, y_input, y_mask, memory, memory_mask):
		'''
		:param y_input: [[bos_id,y1,y2,y3,P,P,P],[bos_id,y1,y2,y3,y4,eos_id,P,P]]
		:param y_mask: [[1,1,1,1,P,P,P],[1,1,1,1,1,P,P]]
		:return:
		'''
		x_hidden = memory[0]
		x_emb = memory[1]
		# [[P,P,y3,bos_id,y1,y2],[P,y4,bos_id,y1,y2,y3]]
		y_pre_target = tf.pad(y_input, paddings=[[0,0],[1,0]], constant_values=0)[:, :-1]
		y_pre_target = self.__reverse(x=y_pre_target)
		# [[0, 0, 0, bos_id, y1, y2, y3], [0, 0, bos_id, y1, y2, y3, y4]]
		y_input = self.__reverse(x=y_input)
		# [[0,0,0,1,1,1,1],[0,0,1,1,1,1,1]]
		y_mask = self.__reverse(x=y_mask)
		# [[P,P,P,bos_id,x1,x2],[P,P,bos_id,x1,x2,x3]]
		y_pre_target = y_pre_target * y_mask

		# batch_size x seq_len x hidden_size
		y_emb = self.__embedding(x_input=y_input,
								 x_mask=y_mask,
								 x_emb_tbl=self._embedding_lookup_tbl_decoder,
								 pos_emb_tbl=self._pos_embedding_lookup_tbl_decoder)


		# batch_size x seq_len x 1 x hidden_size
		y_hidden = tf.expand_dims(input=y_emb, axis=-2)

		for i in range(len(self._config.kernel_list)):
			kernel_size = self._config.kernel_list[i]
			y_hidden_pre = y_hidden

			y_hidden = tf.pad(tensor=y_hidden, paddings=[[0, 0], [kernel_size - 1, 0], [0,0], [0,0]], constant_values=0)
			y1 = tf.nn.conv2d(input=y_hidden, filter=self._kernal_list_decoder[i][0], padding='VALID', strides=[1,1,1,1])
			y2 = tf.nn.conv2d(input=y_hidden, filter=self._kernal_list_decoder[i][1], padding='VALID', strides=[1,1,1,1])

			input_gate = tf.sigmoid(x=y1)
			y_hidden = y2 * input_gate
			y_hidden = y_hidden * y_mask
			# 一个小小的trick，从别人的实现里面抄的
			y_hidden = (y_hidden + y_hidden_pre) * tf.sqrt(0.5)
			# 线性投射并执行attention
			decode_project = self._project_list_decoder[i]
			q = decode_project(y_hidden) + y_pre_target
			q_mask = y_mask
			k = x_hidden
			k_mask = memory_mask
			v = x_hidden + x_emb
			v_mask = memory_mask
			c = scaled_dot_product_attention(q=q, mask_q=q_mask,
										 k=k, mask_k=k_mask,
										 v=v, mask_v=v_mask,
										 is_training=self._mode==tf.estimator.ModeKeys.TRAIN)
			y_hidden = c + y_hidden

		y_hidden = tf.squeeze(input=y_hidden, axis=-2)

		weights = tf.transpose(a=self._embedding_lookup_tbl_decoder)
		logits = tf.einsum('ntd,dk->ntk', y_hidden, weights)
		scores = tf.nn.softmax(logits=logits, axis=-1)

		return logits, scores

	def create_model(self, x_input, y_input):

		x_mask = make_mask_by_value(x=x_input)
		memory, x_emb = self.encode(x_input=x_input, x_mask=x_mask)
		y_mask = make_mask_by_value(x=y_input)
		logtis, scores = self.decode(y_input=y_input, y_mask=y_mask, memory=[memory, x_emb], memory_mask=x_mask)

		return logtis, scores

	def calculate_loss(self, logits, y_labels):

		y_label_mask = make_mask_by_value(x=y_labels)
		log_probs = tf.nn.log_softmax(logits=logits, axis=-1)
		one_hot_labels = tf.one_hot(indices=y_labels, depth=self._config.vocab_size, dtype=tf.float32)
		smoothed_one_hot_labels = super().label_smoothing(inputs=one_hot_labels)
		per_sample_loss = -tf.reduce_sum(input_tensor=(smoothed_one_hot_labels * log_probs), axis=-1)
		per_sample_loss = per_sample_loss * tf.cast(x=y_label_mask, dtype=tf.float32)
		loss = tf.reduce_sum(input_tensor=per_sample_loss) / tf.reduce_sum(tf.cast(x=y_label_mask, dtype=tf.float32))

		return loss

