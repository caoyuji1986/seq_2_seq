import tensorflow as tf
import numpy as np
import time

def data_generator():

    dataset = np.array(range(500))
    for d in dataset:
        yield d

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