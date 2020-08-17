import tensorflow as tf
import numpy as np
import time

def data_generator():

    dataset = np.array(range(500))
    for d in dataset:
        yield d
def main():
    d = tf.data.Dataset.from_generator(generator=data_generator, output_types=tf.int64, output_shapes=[])

    d = d.map(map_func=lambda  x: x+1)

    def key_fn(x):

        return x // 10

    def reduce_fn(k, v):

        return v.batch(20)

    d = d.shuffle(buffer_size=23)
    d = d.apply(tf.data.experimental.group_by_window(key_func=key_fn, reduce_func=reduce_fn, window_size=100))
    d = d.shuffle(buffer_size=23)
    itr = d.make_initializable_iterator()
    x = itr.get_next()
    sess = tf.Session()

    sess.run(itr.initializer)
    while True:
        a = sess.run(x)
        print(a)
        time.sleep(1)

def test_variable_scope():

    initializer = tf.ones_initializer(dtype=tf.float32)
    with tf.variable_scope(name_or_scope="a", initializer=initializer):
        a = tf.get_variable(name="a", shape=[2,3])
        b = tf.convert_to_tensor(value=[[1,2,3],[4,5,6]], dtype=tf.float32)
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=3)
        s = cell.zero_state(batch_size=2, dtype=tf.float32)
        r = cell.__call__(inputs=a+b, state=s)
        c = a + b

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    c_value = sess.run(r)
    print(c_value)

def test_lr_decay():
    def noam_scheme_seq_2_seq(global_step, step_num_in_epoch):
        '''
        论文中总共训练了7.5 个epoch，其中前5个epoch 保持learning rate 不变化为0.7， 后面每增加一个epoch 就 learning rate 变为原来的一半；
        这里我再前7.5个 epoch 使用这种策略，后面保持不变
        '''
        lr = 0.00546875
        f_threshold = tf.constant(step_num_in_epoch * 5.0)
        s_threshold = tf.constant(step_num_in_epoch * 7.5)
        s_value = lr / tf.pow(x=2.0, y=(global_step // step_num_in_epoch - 4))
        e_value = lr / tf.pow(x=2.0, y=(step_num_in_epoch * 7.5 // step_num_in_epoch - 4))

        r = tf.cond(pred=tf.less(x=global_step, y=f_threshold),
                    true_fn=lambda: tf.convert_to_tensor(value=lr, dtype=tf.float32),
                    false_fn=lambda: tf.cond(pred=tf.less(x=global_step, y=s_threshold),
                                     true_fn=lambda: s_value,
                                     false_fn=lambda: e_value))

        return r
    gs = tf.convert_to_tensor(value=1, dtype=tf.float32)
    gs = tf.placeholder(dtype=tf.float32, shape=[])
    r = noam_scheme_seq_2_seq(gs, 10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    r_v = sess.run(r, feed_dict={
        gs:1
    })
    print(r_v)
    r_v = sess.run(r, feed_dict={
        gs: 40
    })
    print(r_v)

    r_v = sess.run(r, feed_dict={
        gs: 51
    })
    print(r_v)

    r_v = sess.run(r, feed_dict={
        gs: 61
    })
    print(r_v)

    r_v = sess.run(r, feed_dict={
        gs: 71
    })
    print(r_v)

    r_v = sess.run(r, feed_dict={
        gs: 75
    })
    print(r_v)

    r_v = sess.run(r, feed_dict={
        gs: 80
    })
    print(r_v)

test_lr_decay()