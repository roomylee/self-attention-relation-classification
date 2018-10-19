import tensorflow as tf

from utils import initializer
from model.attention import multihead_attention


class SelfCNN:
    def __init__(self, sequence_length, num_classes,
                 vocab_size, embedding_size, pos_vocab_size, pos_embedding_size,
                 filter_sizes, num_filters, num_heads, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.input_p1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p1')
        self.input_p2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p2')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Word Embedding Layer
        with tf.device('/cpu:0'), tf.variable_scope("word-embeddings"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25), name="W_text")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_x)

        # Dropout for Word Embedding
        with tf.variable_scope('dropout-embeddings'):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.emb_dropout_keep_prob)

        # Self Attention
        with tf.variable_scope("self-attention"):
            self.self_attn, self.self_alphas = multihead_attention(self.embedded_chars, self.embedded_chars,
                                                                   num_units=embedding_size, num_heads=num_heads)
            self.self_attn_expanded = tf.expand_dims(self.self_attn, -1)

        # Position Embedding Layer
        with tf.device('/cpu:0'), tf.variable_scope("position-embeddings"):
            self.W_pos = tf.get_variable("W_pos", [pos_vocab_size, pos_embedding_size], initializer=initializer())
            self.p1 = tf.nn.embedding_lookup(self.W_pos, self.input_p1)[:, :tf.shape(self.embedded_chars)[1]]
            self.p2 = tf.nn.embedding_lookup(self.W_pos, self.input_p2)[:, :tf.shape(self.embedded_chars)[1]]
            self.p1_expanded = tf.expand_dims(self.p1, -1)
            self.p2_expanded = tf.expand_dims(self.p2, -1)

        self.emb_expanded = tf.concat([self.self_attn_expanded, self.p1_expanded, self.p2_expanded], 2)
        _embedding_size = embedding_size + 2 * pos_embedding_size

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                conv = tf.layers.conv2d(self.emb_expanded, num_filters, [filter_size, _embedding_size],
                                        kernel_initializer=initializer(), activation=tf.nn.relu, name="conv")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(conv, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.variable_scope("output"):
            self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2

        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
