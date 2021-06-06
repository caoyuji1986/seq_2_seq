# coding:utf8

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(name='model_dir', default='./out/model/', help='模型的位置')
flags.DEFINE_string(name='data_dir', default='./out/dat/de_en/', help='模型的位置')
flags.DEFINE_list(name='train_files',
                  default=['./dat/wmt17/de_en/all/corpus.tc.de','./dat/wmt17/de_en/all/corpus.tc.en'],
                  help='训练和测试数据的位置')
flags.DEFINE_list(name='eval_files',
                  default=['./dat/wmt17/de_en/all/newstest2014.tc.de','./dat/wmt17/de_en/all/newstest2014.tc.en'],
                  help='训练和测试数据的位置')
flags.DEFINE_list(name='bpe_model_files',
                  default=['./dat/wmt17/de_en/de.model','./dat/wmt17/de_en/en.model'],
                  help='bpe 模型位置')


flags.DEFINE_integer(name='max_token_num', default=64, help='每个句子的最大分词数量')
flags.DEFINE_integer(name='bucket_num', default=10, help='动态组织batch的时候的分桶数量')

flags.DEFINE_boolean(name='do_predict', default=False, help='是否开始预测')
flags.DEFINE_boolean(name='do_train', default=True, help='是否开始训练')

flags.DEFINE_integer(name='save_checkpoint_steps', default=1000, help='保存ckpt的训练步数')
flags.DEFINE_integer(name='keep_checkpoint_max', default=5, help='最多保存多少ckpt')
flags.DEFINE_string(name='model_config', default='./cfg/transformer.json', help='模型配置位置')

flags.DEFINE_integer(name='num_train_samples', default=5000000, help='训练样本数量')
flags.DEFINE_integer(name='warmup_steps', default=4000, help='预热步数')
flags.DEFINE_integer(name='num_epoches', default=40, help='epoches 的数量')
flags.DEFINE_integer(name='batch_size', default=25600, help='batch size or window size')
flags.DEFINE_integer(name='token_num_in_batch', default=25000, help='每个batch 中的符号数量')

flags.DEFINE_string(name='model_type', default='transformer', help='使用的模型名称')

def main(un_used):
    print(FLAGS.train_files)


if __name__=='''__main__''':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()