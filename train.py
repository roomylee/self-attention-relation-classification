import os
import time
import numpy as np
import tensorflow as tf

import data_helpers
from configure import FLAGS
from logger import Logger
from model.self_lstm import SelfLSTM
from model.self_cnn import SelfCNN
import utils

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def train():
    with tf.device('/cpu:0'):
        train_text, train_y = data_helpers.load_data_and_labels(FLAGS.train_path)
    with tf.device('/cpu:0'):
        test_text, test_y = data_helpers.load_data_and_labels(FLAGS.test_path)

    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = MAX_SENTENCE_LENGTH
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    vocab_processor.fit(train_text + test_text)
    train_x = np.array(list(vocab_processor.transform(train_text)))
    test_x = np.array(list(vocab_processor.transform(test_text)))
    print("\nText Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("train_x = {0}".format(train_x.shape))
    print("train_y = {0}".format(train_y.shape))
    print("test_x = {0}".format(test_x.shape))
    print("test_y = {0}".format(test_y.shape))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if FLAGS.model == 'cnn':
                model = SelfCNN(
                    sequence_length=train_x.shape[1],
                    num_classes=train_y.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_size,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    num_heads=FLAGS.num_heads,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
            elif FLAGS.model == 'lstm':
                model = SelfLSTM(
                    sequence_length=train_x.shape[1],
                    num_classes=train_y.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_size,
                    hidden_size=FLAGS.hidden_size,
                    num_heads=FLAGS.num_heads,
                    attention_size=FLAGS.attention_size,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(model.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("\nWriting to {}\n".format(out_dir))

            # Logger
            logger = Logger(out_dir)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            if FLAGS.embeddings == "word2vec":
                pretrain_W = utils.load_word2vec('resource/GoogleNews-vectors-negative300.bin', FLAGS.embedding_size, vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained word2vec model!\n")
            elif FLAGS.embeddings == "glove100":
                pretrain_W = utils.load_glove('resource/glove.6B.100d.txt', FLAGS.embedding_size, vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained glove100 model!\n")
            elif FLAGS.embeddings == "glove300":
                pretrain_W = utils.load_glove('resource/glove.840B.300d.txt', FLAGS.embedding_size, vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained glove300 model!\n")

            # Generate batches
            train_batches = data_helpers.batch_iter(list(zip(train_x, train_y)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_f1 = 0.0  # For save checkpoint(model)
            for train_batch in train_batches:
                train_bx, train_by = zip(*train_batch)
                feed_dict = {
                    model.input_x: train_bx,
                    model.input_y: train_by,
                    model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                    model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    logger.logging_train(step, loss, accuracy)

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    # Generate batches
                    test_batches = data_helpers.batch_iter(list(zip(test_x, test_y)), FLAGS.batch_size, 1, shuffle=False)
                    # Training loop. For each batch...
                    losses = 0.0
                    accuracy = 0.0
                    predictions = []
                    iter_cnt = 0
                    for test_batch in test_batches:
                        test_bx, test_by = zip(*test_batch)
                        feed_dict = {
                            model.input_x: test_bx,
                            model.input_y: test_by,
                            model.emb_dropout_keep_prob: 1.0,
                            model.rnn_dropout_keep_prob: 1.0,
                            model.dropout_keep_prob: 1.0
                        }
                        loss, acc, pred = sess.run(
                            [model.loss, model.accuracy, model.predictions], feed_dict)
                        losses += loss
                        accuracy += acc
                        predictions += pred.tolist()
                        iter_cnt += 1
                    losses /= iter_cnt
                    accuracy /= iter_cnt
                    predictions = np.array(predictions, dtype='int')

                    logger.logging_eval(step, loss, accuracy, predictions)

                    # Model checkpoint
                    if best_f1 < logger.best_f1:
                        best_f1 = logger.best_f1
                        path = saver.save(sess, checkpoint_prefix+"-{:.3g}".format(best_f1), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
