# coding:utf8

import collections
import linecache
import os
import tensorflow as tf
import sentencepiece as spm


class Example:
	
	def __init__(self, guid, x, y, y_label):
		self.guid = guid
		self.x = x
		self.y = y
		self.y_label = y_label


class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""
	
	def __init__(self, bpe_model_file_src, bpe_model_file_dst):
		self._sent_piece_src = spm.SentencePieceProcessor()
		self._sent_piece_src.Load(bpe_model_file_src)
		self._sent_piece_dst = spm.SentencePieceProcessor()
		self._sent_piece_dst.Load(bpe_model_file_dst)
	
	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		
		x_lines, y_lines = self._read_txt([os.path.join(data_dir, "train.src"), os.path.join(data_dir, "train.dst")])
		return self._create_examples(x_lines=x_lines, y_lines=y_lines, set_type='train')
	
	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()
	
	def get_test_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for prediction."""
		x_lines, y_lines = self._read_txt([os.path.join(data_dir, "eval.src"), os.path.join(data_dir, "eval.dst")])
		return self._create_examples(x_lines=x_lines, y_lines=y_lines, set_type='test')
	
	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()
	
	@classmethod
	def _read_txt(cls, input_files):
		"""Reads a tab separated value file."""
		lines = linecache.getlines(filename=input_files[0])
		x_lines = [line.strip() for line in lines]
		
		lines = linecache.getlines(filename=input_files[1])
		y_lines = [line.strip() for line in lines]
		
		return x_lines, y_lines
	
	def _create_examples(self, x_lines, y_lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for i in range(len(x_lines)):
			x_line = x_lines[i]
			y_line = y_lines[i]
			guid = "%s-%d" % (set_type, i)
			eos_id_src = self._sent_piece_src.eos_id()
			bos_id_dst = self._sent_piece_dst.bos_id()
			eos_id_dst = self._sent_piece_dst.eos_id()
			assert eos_id_src==eos_id_dst
			x = self._sent_piece_src.encode_as_ids(input=x_line)[:1] + [eos_id_src]
			y = [bos_id_dst] + self._sent_piece_dst.encode_as_ids(input=y_line)[:1]
			y_label = y[1:] + [eos_id_dst]
			
			examples.append(
				Example(guid=guid, x=x, y=y, y_label=y_label)
			)
		return examples


def file_based_convert_examples_to_features(examples, output_file):
	"""Convert a set of `InputExample`s to a TFRecord file."""
	
	writer = tf.python_io.TFRecordWriter(output_file)
	
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
		
		def create_int_feature(values):
			f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
			return f
		
		features = collections.OrderedDict()
		features["x"] = create_int_feature(example.x)
		features["y"] = create_int_feature(example.y)
		features["y_label"] = create_int_feature(example.y_label)
		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		writer.write(tf_example.SerializeToString())
	
	writer.close()


def file_based_input_fn_builder(input_file, is_training,
                                drop_remainder):
	"""Creates an `input_fn` closure to be passed to Estimator."""
	
	name_to_features = {
		"x": tf.VarLenFeature(tf.int64),
		"y": tf.VarLenFeature(tf.int64),
		"y_label": tf.VarLenFeature(tf.int64)
	}
	
	def _decode_record(record, name_to_features):
		"""Decodes a record to a TensorFlow example."""
		example = tf.parse_single_example(record, name_to_features)
		
		for name in list(example.keys()):
			t = example[name]
			if t.dtype == tf.int64:
				t = tf.to_int32(t)
				t = tf.sparse_tensor_to_dense(t)
			example[name] = t
		
		return (example['x'], example['y']), (example['y_label'])
	
	def input_fn(params):
		
		"""The actual input function."""
		batch_size = params["train_batch_size"]
		
		# For training, we want a lot of parallel reading and shuffling.
		# For eval, we want no shuffling and parallel reading doesn't matter.
		d = tf.data.TFRecordDataset(input_file)
		if is_training:
			d = d.repeat()
			d = d.shuffle(buffer_size=1000000)
		
		d = d.map(
			map_func=lambda record: _decode_record(record, name_to_features),
			num_parallel_calls=16
		)
		padded_shapes = (
			([None], [None]),
			([None])
		)
		padding_values = (
			(0, 0),(0)
		)
		d = d.padded_batch(batch_size=batch_size,
		                   padded_shapes=padded_shapes,
		                   padding_values=padding_values,
		                   drop_remainder=drop_remainder)
		# 启动数据pipe line
		d = d.prefetch(1)
		return d
	
	return input_fn
