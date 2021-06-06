import tensorflow as tf

memory = tf.convert_to_tensor(value=[[[1,1,1,],[2,2,2]],
                                     [[1,1,1,],[2,2,2]]], dtype=tf.float32)
source_sequence_length=tf.convert_to_tensor(value=[2,2])
attention_mechanism = tf.contrib.seq2seq.LuongAttention(10, memory,
                                                        memory_sequence_length=source_sequence_length)
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=3)
tf.nn.rnn_cell.BasicRNNCell
cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                   attention_layer_size=3,
                                                   output_attention=True,
                                                name='attention')