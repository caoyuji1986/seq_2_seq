#coding:utf-8
from operation import *
import tensorflow as tf
import numpy as np
import six
import copy
import json
import math


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


class Transformer:
	
	
	def __init__(self, config, mode):
		"""
		:type config: TransformerConfig
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
			embedding_lookup_tbl = tf.get_variable(name='embedding_lookup_tbl',
																								 shape=[self._config.vocab_size, self._config.hidden_size],
																								 dtype=tf.float32,
																								 initializer=tf.contrib.layers.xavier_initializer())
			zero_emb = tf.zeros(shape=[1, self._config.hidden_size], dtype=tf.float32)
			self._embedding_lookup_tbl = tf.concat(values=[zero_emb, embedding_lookup_tbl[1:]], axis=0)

		with tf.variable_scope(name_or_scope='position_embedding_tbl', reuse=tf.AUTO_REUSE):
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
				beta1 = tf.get_variable("layer_norm_1_beta" + str(layer_index), self._config.hidden_size,
				                       initializer=tf.zeros_initializer())
				gamma1 = tf.get_variable("layer_norm_1_gamma" + str(layer_index), self._config.hidden_size,
				                        initializer=tf.ones_initializer())
				layer_norm_1 =(beta1, gamma1)
				feed_forward_inner = DenseOpt(src_dim=self._config.hidden_size, dst_dim=self._config.position_wise_feed_forward_size,
				                        active_fn=tf.nn.relu, name='feed_forward_inner' + str(layer_index))
				feed_forward_outer = DenseOpt(src_dim=self._config.position_wise_feed_forward_size, dst_dim=self._config.hidden_size,
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
				linear_project = None
				if layer_index == self._config.num_hidden_layers - 1:
					linear_project = DenseOpt(src_dim=self._config.hidden_size, dst_dim=self._config.vocab_size, name='linear_project')
				self._decoder_opt.append({
					'att1': att1,
					'layer_norm_1': layer_norm_1,
					'att2': att2,
					'layer_norm_2': layer_norm_2,
					'feed_forward': feed_forward,
					'layer_norm_3': layer_norm_3,
					'linear_project': linear_project
				})
		self._mode = mode
				
			
	def encode(self, x_input, x_mask):
		
		x_id_emb = tf.nn.embedding_lookup(params=self._embedding_lookup_tbl, ids=x_input)
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

		return x
		
	
	def decode(self, y_input, y_mask, memory, memory_mask):

		y_id_emb = tf.nn.embedding_lookup(params=self._embedding_lookup_tbl, ids=y_input)
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
			                                                    mask_k=y_mask, mask_q=memory_mask, mask_v=memory_mask,
			                                                    attention_dropout=self._config.sub_layer_dropout_prob,
			                                                    is_training=self._mode == tf.estimator.ModeKeys.TRAIN,
			                                                    dk=self._config.attention_size, attention_future=False
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
		
		logits = self._decoder_opt[-1]['linear_project'](x_input=y)
		scores = tf.nn.softmax(logits=logits, axis=-1)
		
		return logits, scores