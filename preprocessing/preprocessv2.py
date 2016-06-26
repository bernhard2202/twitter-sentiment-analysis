#!/usr/bin/env python3
import pickle
import gensim
import getopt
import numpy as np
import sys

FULL_POS_FILE_NAME = "../data/train/train_pos_full.txt"
FULL_NEG_FILE_NAME = "../data/train/train_neg_full.txt"
POS_FILE_NAME = "../data/train/train_pos.txt"
NEG_FILE_NAME = "../data/train/train_neg.txt"
VALID_FILE_NAME = "../data/test/test_data.txt"
VOCAB_FILE_NAME = "../data/preprocessing/vocab_cut.txt"
WORD2VEC_FILE_NAME = "../data/word2vec/GoogleNews-vectors-negative300.bin"
MAPPINGS_FOLDER = "../data/preprocessing/mappings/"

VALID_SIZE = 10000
FULL_TRAIN_SIZE = 2500000
SMALL_TRAIN_SIZE = 200000


with open(MAPPINGS_FOLDER+"mappings.pkl", 'rb') as f:
    (mappings, pretrained, extra_words) = pickle.load(f)


def vocab_and_embeddings(prefix):
    """
    pickle vocabulary
    """
    vocab = dict()
    vocab_inv = dict()
    # pre insert the padding word
    index = 0
    vocab["<PAD/>"] = index
    vocab_inv[index] = "<PAD/>"
    index += 1

    print("We have {0} extra words.".format(len(extra_words)))
    for word in extra_words:
        if word in pretrained:
            print("in extra_words and pretrained simultaneously!: "+word)

    for word in extra_words:
        vocab[word] = index
        vocab_inv[index] = word
        index += 1

    """
    pickle word embeddings
    """
    model = gensim.models.word2vec.Word2Vec.load_word2vec_format(WORD2VEC_FILE_NAME, binary=True)
    embedding_dim = len(model['queen'])
    X = np.empty( (len(extra_words)+len(pretrained)+1  ,embedding_dim)  )
    X[0:len(extra_words)+1] = np.random.uniform(-0.25, 0.25, size=(len(extra_words)+1, embedding_dim))
    print("create word2vec lookup table..")

    assert index == len(extra_words)+1

    for word in pretrained:
        vocab[word] = index
        vocab_inv[index] = word
        X[index] = model[word]
        index += 1

    # sanity check
    print("len(vocab)= {}".format(len(vocab)))
    print("len(vocab_inv)= {}".format(len(vocab_inv)))
    print("len(extra_words)+len(pretrained)+1= {}".format(len(extra_words)+len(pretrained)+1))
    #assert len(vocab) == (len(extra_words) + len(pretrained) + 1) == len(vocab_inv)

    with open('../data/preprocessing/{0}-vocab.pkl'.format(prefix), 'wb') as f:
        pickle.dump(vocab, f, protocol=2)
    with open('../data/preprocessing/{0}-vocab-inv.pkl'.format(prefix), 'wb') as f:
        pickle.dump(vocab_inv, f, protocol=2)
    print("Vocabulary pickled.")

    np.save('../data/preprocessing/{0}-embeddings'.format(prefix), X)
    print("Embeddings pickled.")
    print("Used {} pre-trained word2vec vectors and {} new random vectors.".format(len(pretrained), len(extra_words)+1))

    return vocab


def handle_hashtags_and_mappings(line, vocab):
    result_line = []
    for word in line.split():
        if word[0] == '#' and word not in vocab:  # if hashtag but is in vocab then leave it as it is (it has big enough frequency)
            word = word[1:]                       # otherwise split it to normal words
            length = len(word)
            word_result = []
            claimed = np.full(length, False, dtype=bool)  #initially all letters are free to select
            for n in range(length, 0, -1):  #initially search for words with n letters, then n-1,... until 1 letter words
                for s in range(0, length-n+1):  #starting point. so we examine substring  [s,s+n)
                    substring = word[s:s+n]
                    if substring in vocab:
                        if ~np.any(claimed[s:s+n]):   #nothing is claimed so take it
                            claimed[s:s+n] = True
                            word_result.append((s, substring))
            word_result.sort()
            for _, substring in word_result:
                result_line.append(substring)
        else:  # it is not a hashtag. check if it has a mapping (spelling correction)
            if word in mappings:
                result_line.append(mappings[word])
            else:
                result_line.append(word)
    return ' '.join(result_line)



def test_preprocessing():
    with open('../data/preprocessing/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    train_pos_file = "../data/train/train_pos_full.txt"
    train_neg_file = "../data/train/train_neg_full.txt"
    max_sentence_length = 35
    train_size = SMALL_TRAIN_SIZE = 200000
    train_X = np.zeros((train_size, max_sentence_length))
    train_Y = np.zeros((train_size, 2))
    # sanity check because we initialize with zero then we don't have to do padding
    assert vocab['<PAD/>'] == 0
    i = 0
    pos = 0
    cut = 0
    empty = 0
    for filename in [train_neg_file]:
        with open(filename) as f:
            cnt = 0
            for line in f:
                cnt += 1
                if cnt > 300:
                    break
                line = line.strip()
                #print("before preprocessing: {}".format(line))
                line = handle_hashtags_and_mappings(line, vocab)
                print(line)


def prepare_data(train_pos_file, train_neg_file, train_size, vocab, max_sentence_length):
    """
    prepare training data
    """
    train_X = np.zeros((train_size, max_sentence_length))
    train_Y = np.zeros((train_size, 2))
    # sanity check because we initialize with zero then we don't have to do padding
    assert vocab['<PAD/>'] == 0
    i = 0    # sentence counter
    pos = 0
    cut = 0
    empty = 0
    print("prepare_data: len(vocab) = {0}".format(len(vocab)))
    print("Train neg file: {0}; Train pos file: {1}".format(train_neg_file,
                                                            train_pos_file))
    for filename in [train_neg_file, train_pos_file]:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                j = 0   # word counter (inside sentence)
                #print("before handle_hashtags: {}".format(line))
                line = handle_hashtags_and_mappings(line, vocab)
                #print("after handle_hashtags: {}".format(line))
                for word in line.split():
                    if word in vocab:
                        train_X[i, j] = vocab[word]
                        j += 1
                    if j == max_sentence_length:
                        cut += 1
                        # print("cut: "+line)
                        # cut sentences longer than max sentence lenght
                        break
                if j == 0:
                    empty += 1
                    # print("empty: "+line)
                if filename in (FULL_POS_FILE_NAME, POS_FILE_NAME):
                    train_Y[i, 0] = 0
                    train_Y[i, 1] = 1
                    pos += 1
                else:
                    train_Y[i, 1] = 0
                    train_Y[i, 0] = 1

                i += 1

    assert pos == (len(train_Y) / 2)
    assert train_Y.shape[0] == train_X.shape[0] == i

    print("{} tweets cut to max sentence lenght and {} tweets disapeared due to filtering."
          .format(cut, empty))
    print("Train X shape: {0}".format(train_X.shape))
    print("Train y shape: {0}".format(train_Y.shape))
    return train_X, train_Y


def prepare_valid_data(max_sentence_length, vocab):
    validate_x = np.zeros((VALID_SIZE, max_sentence_length))
    i = 0
    cut = 0
    empty = 0
    with open(VALID_FILE_NAME) as f:
        for tweet in f:
            tweet = tweet.strip()
            tweet = tweet[6:]   # remove prefix   "<num>,"
            tweet = handle_hashtags_and_mappings(tweet, vocab)
            j = 0
            for word in tweet.split():
                if word in vocab:
                    validate_x[i, j] = vocab[word]
                    j += 1
                if j == max_sentence_length:
                    cut += 1
                    # print("cut: "+line)
                    # cut sentences longer than max sentence lenght
                    break
            if j == 0:
                #print(tweet)
                empty += 1
            i += 1
    print("Preprocessing done. {} tweets cut to max sentence lenght and {} tweets disapeared due to filtering."
          .format(cut, empty))
    return validate_x



def usage():
    print("usage: preprocessv2.py  [--full] [--sentence-length=]")
    print("\t--full use full data set")
    print("\t--sentence-length= maximum sentence length")


def main(argv):
    max_sentence_length = 30
    train_pos_file = POS_FILE_NAME
    train_neg_file = NEG_FILE_NAME
    train_size = SMALL_TRAIN_SIZE
    prefix = 'subset'

    try:
        opts, args = getopt.getopt(argv, "[fl:]", ["full", "sentence-length="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        # TODO(andrei): Warn when not using full subset!
        if opt in ("-f", "--full"):
            train_pos_file = FULL_POS_FILE_NAME
            train_neg_file = FULL_NEG_FILE_NAME
            train_size = FULL_TRAIN_SIZE
            prefix = 'full'
            print("train in Full data")
        if opt in ("-s", "--sentence-length"):
            max_sentence_length = int(arg)

    print("sentence-length={}".format(max_sentence_length))
    print('Pickle vocabulary and build word embeddings (this can take some time)..')
    vocab = vocab_and_embeddings(prefix)

    print('prepare training data..')
    X, Y = prepare_data(train_pos_file, train_neg_file, train_size, vocab, max_sentence_length)

    print("Will save with prefix: {0}".format(prefix))
    np.save('../data/preprocessing/{0}-trainX'.format(prefix), X)
    np.save('../data/preprocessing/{0}-trainY'.format(prefix), Y)

    print('prepare validation data..')
    validate_x = prepare_valid_data(max_sentence_length, vocab)
    np.save('../data/preprocessing/validateX', validate_x)


if __name__ == "__main__":
    main(sys.argv[1:])
    #test_preprocessing()

# ./preprocessv2.py --full --sentence-length=35
