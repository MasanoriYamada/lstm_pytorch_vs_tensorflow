from data.wavyreaching2d100 import Loader
import numpy as np
import tensorflow as tf

# Input size
nin = 2
# Hidden layer num
n_hidden = 256
# Sequence max length
seq_max_len = 100
bs = 80


# #batch_size * #max_seq_len * #inputsize
input = tf.placeholder(tf.float32, [None, seq_max_len, nin])
# #batch_size
sequence_length = tf.placeholder(tf.int32, [None])
batch_size = tf.shape(input)[0]

# model
weights_initializer = tf.initializers.ones
cell = tf.nn.rnn_cell.LSTMCell(n_hidden, initializer=weights_initializer, forget_bias=1, activation=tf.nn.sigmoid)
initial_state = cell.zero_state(batch_size, dtype=tf.float32)

output, state = tf.nn.dynamic_rnn(
    cell, input, initial_state=initial_state,
    sequence_length=sequence_length)


# Sequence length
seqlen = np.array([seq_max_len]*bs)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    data_loader = Loader(bs)
    train_loader = data_loader.get_train_loader()

    for i, (x_seg_id, label) in enumerate(train_loader):
        x = x_seg_id[0].numpy()  # batch t, f_dim
        y = x_seg_id[1].numpy()  # seg_id
        
        out, st = sess.run([output, state], {input: x, sequence_length: seqlen})
        h = st.h
        c = st.c

        print(h)
        print(c)
        print(out)
        
print('end')

