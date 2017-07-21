import datetime

import tensorflow as tf

from cnn.tensorflow.insurance_qa_data_helpers import build_initial_embedding_matrix


##########################################################################
#  embedding_lookup + cnn + cosine margine ,  batch
##########################################################################


class InsQACNN(object):
    def __init__(
            self, sequence_length, batch_size,
            d_word2idx, embedding_size,
            filter_sizes, num_filters, glove_path):
        # 用户问题,字向量使用embedding_lookup
        self.input_x_1 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x_1")
        # 待匹配正向问题
        self.input_x_2 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x_2")
        # 负向问题
        self.input_x_3 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x_3")
        print("input_x_1 ", self.input_x_1)

        # Initalize the word vector with Glove
        vocab_size = len(d_word2idx)
        initializer = build_initial_embedding_matrix(d_word2idx, glove_path, embedding_size)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.get_variable(
                "W",
                shape=[vocab_size, embedding_size],
                trainable=True,
                initializer=initializer)
            chars_1 = tf.nn.embedding_lookup(W, self.input_x_1)
            chars_2 = tf.nn.embedding_lookup(W, self.input_x_2)
            chars_3 = tf.nn.embedding_lookup(W, self.input_x_3)
            self.embedded_chars_1 = chars_1
            self.embedded_chars_2 = chars_2
            self.embedded_chars_3 = chars_3
        self.embedded_chars_expanded_1 = tf.expand_dims(self.embedded_chars_1, -1)
        self.embedded_chars_expanded_2 = tf.expand_dims(self.embedded_chars_2, -1)
        self.embedded_chars_expanded_3 = tf.expand_dims(self.embedded_chars_3, -1)

        pooled_outputs_1 = []
        pooled_outputs_2 = []
        pooled_outputs_3 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_1,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-1")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-1"
                )
                pooled_outputs_1.append(pooled)

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_2,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-2"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-2")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-2"
                )
                pooled_outputs_2.append(pooled)

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_3,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-3"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-3")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-3"
                )
                pooled_outputs_3.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        pooled_reshape_1 = tf.reshape(tf.concat(pooled_outputs_1, axis=3), [-1, num_filters_total])
        pooled_reshape_2 = tf.reshape(tf.concat(pooled_outputs_2, axis=3), [-1, num_filters_total])
        pooled_reshape_3 = tf.reshape(tf.concat(pooled_outputs_3, axis=3), [-1, num_filters_total])

        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_reshape_1, pooled_reshape_1), 1))  # 计算向量长度Batch模式
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_reshape_2, pooled_reshape_2), 1))
        pooled_len_3 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_reshape_3, pooled_reshape_3), 1))
        pooled_mul_12 = tf.reduce_sum(tf.multiply(pooled_reshape_1, pooled_reshape_2), 1)  # 计算向量的点乘Batch模式
        pooled_mul_13 = tf.reduce_sum(tf.multiply(pooled_reshape_1, pooled_reshape_3), 1)

        with tf.name_scope("output"):
            self.cos_12 = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2), name="scores")  # 计算向量夹角Batch模式
            self.cos_13 = tf.div(pooled_mul_13, tf.multiply(pooled_len_1, pooled_len_3))

        zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
        margin = tf.constant(0.05, shape=[batch_size], dtype=tf.float32)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(self.cos_12, self.cos_13)))
            self.loss = tf.reduce_sum(self.losses)
            print('loss ', self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(zero, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")

        # Define Training procedure
        self.optimizer = tf.train.AdamOptimizer(1e-1)
        self.train_op = self.optimizer.minimize(self.loss)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        # Dev summaries
        self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        self.saver = tf.train.Saver(tf.global_variables())

    def train_step(self, x_batch_1, x_batch_2, x_batch_3, sess, step):
        """
        A single training step
        """
        feed_dict = {
            self.input_x_1: x_batch_1,
            self.input_x_2: x_batch_2,
            self.input_x_3: x_batch_3
        }
        _, loss, accuracy = sess.run(
            [self.train_op, self.loss, self.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

    def dev_step(self, x_test_1, x_test_2, x_test_3, sess):
        feed_dict = {
            self.input_x_1: x_test_1,
            self.input_x_2: x_test_2,
            self.input_x_3: x_test_3
        }
        batch_scores = sess.run([self.cos_12], feed_dict)
        return batch_scores
