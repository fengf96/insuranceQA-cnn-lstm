import random

import numpy as np
import tensorflow as tf


def build_vocab(train_file, valid_file):
    code = int(0)
    vocab = {}
    vocab['UNKNOWN'] = code
    code += 1
    for line in open(train_file):
        items = line.strip().split(' ')
        for i in range(2, 4):
            words = items[i].split('_')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    for line in open(valid_file):
        items = line.strip().split(' ')
        for i in range(2, 4):
            words = items[i].split('_')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    return vocab


def rand_qa(qalist):
    index = random.randint(0, len(qalist) - 1)
    return qalist[index]


def read_alist_answers(train_file):
    alist = []
    for line in open(train_file):
        items = line.strip().split(' ')
        alist.append(items[3])
    print('read_alist done ......')
    return alist


def load_test(val_file):
    testList = []
    for line in open(val_file):
        testList.append(line.strip())
    return testList


def read_raw(train_file):
    raw = []
    for line in open(train_file):
        items = line.strip().split(' ')
        if items[0] == '1':
            raw.append(items)
    return raw


def encode_sent(vocab, string, size=200):
    x = []
    words = string.split('_')
    for i in range(0, size):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x


def load_data_6(vocab, alist_answers, raw, size):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, size):
        items = raw[random.randint(0, len(raw) - 1)]
        nega = rand_qa(alist_answers)
        x_train_1.append(encode_sent(vocab, items[2]))
        x_train_2.append(encode_sent(vocab, items[3]))
        x_train_3.append(encode_sent(vocab, nega))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)


def load_data_val_6(testList, vocab, index, batch):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, batch):
        true_index = index + i
        if (true_index >= len(testList)):
            true_index = len(testList) - 1
        items = testList[true_index].split(' ')
        x_train_1.append(encode_sent(vocab, items[2]))
        x_train_2.append(encode_sent(vocab, items[3]))
        x_train_3.append(encode_sent(vocab, items[3]))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


####################################################
# load vocab and glove
####################################################

def load_glove_vectors(glove_path, d_word2idx, initial_embeddings):
    """
    Load glove vectors from a .txt file.
    Optionally limit the vocabulary to save memory. `vocab` should be a set.
    """
    oov = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            tokens = line.split(" ")
            word = tokens[0]
            entries = tokens[1:]
            if not d_word2idx or word in d_word2idx:
                current_idx = d_word2idx[word]
                vec = np.array([float(x) for x in entries])
                initial_embeddings[current_idx, :] = vec
            else:
                oov += 1
        tf.logging.info("Found {} out of {} vectors in Glove".format(oov, len(d_word2idx)))
        return initial_embeddings


def build_initial_embedding_matrix(d_word2idx, glove_path, embedding_size):
    initial_embeddings = np.random.uniform(-0.25, 0.25, (len(d_word2idx), embedding_size)).astype("float32")
    initial_embeddings = load_glove_vectors(glove_path, d_word2idx, initial_embeddings)
    return tf.constant_initializer(initial_embeddings)
