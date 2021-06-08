# coding:utf8

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(name='model_dir', default='./out/model/transformer/', help='模型的位置')
flags.DEFINE_string(name='data_dir', default='./dat/wnt17/en_de/', help='训练和测试数据的位置')
flags.DEFINE_list(name='train_files', default=['corpus.tc.en', 'corpus.tc.de'], help='训练文件')
flags.DEFINE_list(name='eval_files', default=['newstest2016.tc.en', 'newstest2016.tc.de'], help='评估文件')
flags.DEFINE_string(name='bpe_model_file', default='./dat/wnt17/bpe.model', help='bpe 模型位置')

flags.DEFINE_integer(name='max_token_num', default=128, help='每个句子的最大分词数量')
flags.DEFINE_integer(name='bucket_num', default=10, help='动态组织batch的时候的分桶数量')

flags.DEFINE_boolean(name='do_predict', default=False, help='是否开始预测')
flags.DEFINE_boolean(name='do_train', default=True, help='是否开始训练')

flags.DEFINE_integer(name='save_checkpoint_steps', default=1000, help='保存ckpt的训练步数')
flags.DEFINE_integer(name='keep_checkpoint_max', default=5, help='最多保存多少ckpt')
flags.DEFINE_string(name='model_config', default='./cfg/transformer.json', help='模型配置位置')

flags.DEFINE_integer(name='num_train_samples', default=5000000, help='训练样本数量')
flags.DEFINE_integer(name='warmup_steps', default=4000, help='预热步数')
flags.DEFINE_integer(name='num_epoches', default=40, help='epoches 的数量')
flags.DEFINE_integer(name='batch_size', default=256, help='batch size')
flags.DEFINE_integer(name='token_in_batch', default=10000, help='每个batch中的符号数')

flags.DEFINE_enum(name='model_type', default='transformer', enum_values=['transformer', 'rnn', 'cnn'], help='使用的模型名称')