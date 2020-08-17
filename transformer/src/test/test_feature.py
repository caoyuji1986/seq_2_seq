import time
import tensorflow as tf

from feature import file_based_input_fn_builder

input_fn = \
    file_based_input_fn_builder(input_file='./dat/iwslt2016/de_en/train.tfrecord',
                                is_training=False, drop_remainder=False, bucket_num=20)
d = input_fn({'train_batch_size': 8})
iter = tf.data.Iterator.from_structure(d.output_types, d.output_shapes)
xs, ys = iter.get_next()
test_init_op = iter.make_initializer(d)

with tf.Session() as sess:
    sess.run(test_init_op)

    while True:

        ret_x, ret_y, ret_yl = sess.run([xs[0], xs[1], ys])
        print("-------------------------------------")
        print(str(ret_x))
        print(str(ret_y))
        print(str(ret_yl))

        time.sleep(1)

