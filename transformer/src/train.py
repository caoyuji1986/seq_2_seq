#coding:utf-8
import tensorflow as tf

import os

import tensorflow as tf

from feature import file_based_input_fn_builder, DataProcessor, file_based_convert_examples_to_features
from flag_center import FLAGS
from model import Transformer, TransformerConfig, RNNTransformer, RNNTransformerConfig


def create_input_fn(input_file, is_training, drop_remainder):
	input_fn = file_based_input_fn_builder(input_file=input_file,
	                                       is_training=is_training,
	                                       drop_remainder=drop_remainder)
	
	return input_fn


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
	'''Noam scheme learning rate decay
	init_lr: initial learning rate. scalar.
	global_step: scalar.
	warmup_steps: scalar. During warmup_steps, learning rate increases
	until it reaches init_lr.
	'''
	step = tf.cast(global_step + 1, dtype=tf.float32)
	return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def noam_scheme_ori(d_model, global_step, warmup_steps=4000.):
	"""
	the decay strategy of paper
	:return:
	"""
	d_model = tf.cast(x=d_model, dtype=tf.float32)
	learning_rate = 1.0 / tf.sqrt(x=d_model) * \
	                tf.minimum(x=1.0 / tf.sqrt(x=global_step), y=global_step / (warmup_steps ** 1.5))
	return learning_rate


def create_train_opt(loss, d_model=512, warmup_steps=4000.0):
	global_steps_ = tf.train.get_or_create_global_step()
	global_step = tf.cast(x=global_steps_, dtype=tf.float32)
	learning_rate = noam_scheme(0.0003, global_step)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.minimize(loss=loss, global_step=global_steps_)
	tf.summary.scalar('learning_rate', learning_rate)
	summaries = tf.summary.merge_all()
	return train_op, learning_rate


def my_model_fn(features, labels, mode, params):
	warmup_steps = min(params['warmup_steps'], params['train_steps'] * 0.1)
	config = params['config']
	x, y = features
	y_label = labels
	if FLAGS.model_type=='transformer':
		transformer = Transformer(config=config, mode=mode)
	else:
		transformer = RNNTransformer(config=config, mode=mode)
	logits, predicts = transformer.create_model(x_input=x, y_input=y)
	loss = transformer.calculate_loss(logits=logits, y_labels=y_label)
	
	if mode == tf.estimator.ModeKeys.TRAIN:
		train_op, learning_rate = create_train_opt(loss=loss,
		                                           d_model=config.hidden_size,
		                                           warmup_steps=warmup_steps)
		hook_dict = {
			'loss': loss,
			'learning_rate': learning_rate
		}
		hook = tf.train.LoggingTensorHook(
			hook_dict,
			every_n_iter=10
		)
		return tf.estimator.EstimatorSpec(
			mode=mode,
			training_hooks=[hook],
			loss=loss,
			train_op=train_op)
	
	elif mode == tf.estimator.ModeKeys.PREDICT:
		
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions={'prediction': predicts}
		)
	
	else:
		
		raise NotImplementedError('not implemented')


def main(unused_params):
	
	train_steps = FLAGS.num_train_samples * FLAGS.num_epoches / FLAGS.batch_size
	tf.logging.info('train steps is %d' % train_steps)
	tf.logging.info(str(FLAGS.flag_values_dict()))
	
	run_config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
	                                    save_checkpoints_steps=FLAGS.save_checkpoint_steps,
	                                    keep_checkpoint_max=FLAGS.keep_checkpoint_max)
	if FLAGS.model_type == 'transformer':
		model_config = TransformerConfig.from_json_file(FLAGS.model_config)
	else:
		model_config = RNNTransformerConfig.from_json_file(FLAGS.model_config)
	tf.logging.info(model_config.to_json_string())
	params = {
		'warmup_steps': FLAGS.warmup_steps,
		'train_steps': train_steps,
		'config': model_config,
		'train_batch_size': FLAGS.batch_size,
		'predict_batch_size': FLAGS.batch_size
	}
	estimator = tf.estimator.Estimator(model_dir=FLAGS.model_dir,
	                                   model_fn=my_model_fn,
	                                   config=run_config,
	                                   params=params)
	data_processor = DataProcessor(bpe_model_file_src=FLAGS.bpe_model_file+'.src',
	                               bpe_model_file_dst=FLAGS.bpe_model_file+'.dst')
	
	if FLAGS.do_train:
		tf_path = os.path.join(FLAGS.data_dir, 'train.tfrecord')
		if os.path.exists(tf_path) == False:
			examples = data_processor.get_train_examples(data_dir=FLAGS.data_dir)
			file_based_convert_examples_to_features(examples=examples, output_file=tf_path)
		tf.logging.info('开始训练transformer')
		train_input_fn = create_input_fn(input_file=tf_path, is_training=True, drop_remainder=False)
		estimator.train(input_fn=train_input_fn, max_steps=train_steps)
	
	elif FLAGS.do_predict:
		tf_path = os.path.join(FLAGS.data_dir, 'test.tfrecord')
		examples = data_processor.get_test_examples(data_dir=FLAGS.data_dir)
		file_based_convert_examples_to_features(examples=examples, output_file=tf_path)
		tf.logging.info('开始使用transformer 进行predict')
		predict_input_fn = create_input_fn(input_file=tf_path, is_training=False, drop_remainder=False)
		result = estimator.predict(input_fn=predict_input_fn)
	
	else:
		raise NotImplementedError('其他模式没有实现')


if __name__ == '''__main__''':
	
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()
