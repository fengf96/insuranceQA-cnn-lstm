#! /usr/bin/env python3.4

import operator
import os

import tensorflow as tf
# from cnn.tensorflow import insurance_qa_data_helpers
from cnn.tensorflow.insqa_cnn import InsQACNN
from cnn.tensorflow.insurance_qa_data_helpers import read_raw, read_alist_answers, build_vocab, load_test, \
    load_data_val_6, load_data_6

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")

# Train/Test/Vocabulary file
tf.flags.DEFINE_string("train_file", '../../insuranceQA/train', "Training file")
tf.flags.DEFINE_string("val_file", '../../insuranceQA/test1', "Validation file")
tf.flags.DEFINE_string("precision_file", '../../insuranceQA/test1.acc', "Precision file")
tf.flags.DEFINE_string("glove_path", "../../insuranceQA/glove.6B.100d.txt", "glove_path")

# Training parameters
tf.flags.DEFINE_integer("seq_length", 200, "Sequence Length (default: 200)")
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_steps", 5000000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 6000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 6000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================


def evaluate_model(model, session):
    testList = load_test(FLAGS.val_file)
    scoreList = []
    i = int(0)
    while True:
        x_test_1, x_test_2, x_test_3 = load_data_val_6(testList, vocab, i,
                                                       FLAGS.batch_size)
        batch_scores = model.dev_step(x_test_1, x_test_2, x_test_3, session)
        for score in batch_scores[0]:
            scoreList.append(score)
        i += FLAGS.batch_size
        if i >= len(testList):
            break
    sessdict = {}
    index = int(0)
    for line in open(FLAGS.val_file):
        items = line.strip().split(' ')
        qid = items[1].split(':')[1]
        if not qid in sessdict:
            sessdict[qid] = []
        sessdict[qid].append((scoreList[index], items[0]))
        index += 1
        if index >= len(testList):
            break
    lev1 = float(0)
    lev0 = float(0)
    of = open(FLAGS.precision_file, 'a')
    for k, v in sessdict.items():
        v.sort(key=operator.itemgetter(0), reverse=True)
        score, flag = v[0]
        if flag == '1':
            lev1 += 1
        if flag == '0':
            lev0 += 1
    of.write('lev1:' + str(lev1) + '\n')
    of.write('lev0:' + str(lev0) + '\n')
    print('lev1 ' + str(lev1))
    print('lev0 ' + str(lev0))
    print("top-1 accuracy: %s" % (lev1 * 1.0 / (lev1 + lev0)))
    of.close()


def create_model(session):
    model = InsQACNN(
        sequence_length=FLAGS.seq_length,
        batch_size=FLAGS.batch_size,
        d_word2idx=vocab,
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        glove_path=FLAGS.glove_path
    )
    print("Writing to {}\n".format(out_dir))

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("model file not loaded correctly. Start fresh new model")
        # Initialize all variables
        session.run(tf.global_variables_initializer())
    return model


# Training
# ==================================================
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
# Checkpoint directory. tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Load data
print("Loading data...")

vocab = build_vocab(FLAGS.train_file, FLAGS.val_file)
vocab_size = len(vocab)
alist_answers = read_alist_answers(FLAGS.train_file)
raw = read_raw(FLAGS.train_file)

print("Load done...")

with tf.device("/gpu:0"):
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        # device_count={'GPU': 0} uncomment if you have a GPU but don't want to use it
    )
    session_conf.gpu_options.allow_growth = True

    with tf.Session(config=session_conf) as sess:
        model = create_model(sess)

        print("start...")

        # Generate batches
        # Training loop. For each batch...
        for n_step in range(FLAGS.num_steps):
            try:
                x_batch_1, x_batch_2, x_batch_3 = load_data_6(vocab, alist_answers, raw,
                                                              FLAGS.batch_size)
                model.train_step(x_batch_1, x_batch_2, x_batch_3, sess, n_step)
                if (n_step + 1) % FLAGS.checkpoint_every == 0:
                    path = model.saver.save(sess, checkpoint_dir, global_step=n_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if (n_step + 1) % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    evaluate_model(model, sess)
                    print("")

            except Exception as e:
                print(e)
