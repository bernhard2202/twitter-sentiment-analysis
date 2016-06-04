#!/usr/bin/env python3
import pickle
import gensim
import getopt
import numpy as np
import sys
#from stop_words import get_stop_words

#stopwords = set(get_stop_words('en'))

FULL_POS_FILE_NAME = "../data/train/train_pos_full.txt"
FULL_NEG_FILE_NAME = "../data/train/train_neg_full.txt"
POS_FILE_NAME = "../data/train/train_pos.txt"
NEG_FILE_NAME = "../data/train/train_neg.txt"
VALID_FILE_NAME = "../data/test/test_data.txt"
VOCAB_FILE_NAME = "../data/preprocessing/vocab_cut.txt"
WORD2VEC_FILE_NAME = "../data/word2vec/GoogleNews-vectors-negative300.bin"

VALID_SIZE = 10000
FULL_TRAIN_SIZE = 2500000
SMALL_TRAIN_SIZE = 200000


def word_filter(word):
    word = ''.join(i for i in word if not i.isdigit())
    if len(word) < 2:
        return None
    # remove hashtags in beginning
    if word[0] == '#':
        return None
    #    word = word[1:]

    # stopwords
    # if word in stopwords:
    #    return None

    # remove numbers
    # if numberPattern.match(word) is not None:
    #    return None

    # remove single chars

    return word


def filter_with_voc(word, voc):
    word = word_filter(word)
    if (word is not None) and (word in voc):
        return word
    else:
        return None


def pickle_vocab():
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
    i = 1
    words = set()

    with open(VOCAB_FILE_NAME) as f:
        for idx, line in enumerate(f):
            word = word_filter(line.strip())
            # word not filtered in preprocessing and word
            # unique after filtering
            i += 1
            if (word is not None) and (word not in words):
                words.add(word)
                vocab[word] = index
                vocab_inv[index] = word
                index += 1

    # sanity check
    assert len(vocab) == (len(words) + 1) == len(vocab_inv)

    with open('../data/preprocessing/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, protocol=2)
    with open('../data/preprocessing/vocab-inv.pkl', 'wb') as f:
        pickle.dump(vocab_inv, f, protocol=2)
    print("Vocabulary pickled.")
    print("Total number of unique words = {}; words filterd by preprocessing = {}".format(len(vocab), (i - index)))

    return vocab


def pickle_word_embeddings(vocab):
    """
    pickle word embeddings
    """
    model = gensim.models.word2vec.Word2Vec.load_word2vec_format(WORD2VEC_FILE_NAME, binary=True)
    # todo more elegant way
    embedding_dim = len(model['queen'])
    changed = 0
    X = np.random.uniform(-0.25, 0.25, size=(len(vocab), embedding_dim))
    print("create word2vec lookup table..")
    for word in vocab:
        if word in model:
            changed += 1
            X[vocab[word]] = model[word]
    np.save('../data/preprocessing/embeddings', X)
    print("Embeddings pickled.")
    print("Used {} pre-trained word2vec vectors and {} new random vectors.".format(changed, (len(vocab) - changed)))



def handle_hashtags(line,vocab):
    result_line = []
    for word in line.split():
        if word[0] == '#':
            word = word[1:]
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
        else:
            result_line.append(word)
    return ' '.join(result_line)

def test_handle_hashtags():
    with open('../data/preprocessing/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    train_pos_file = "../data/train/train_pos.txt"
    train_neg_file = "../data/train/train_neg.txt"
    max_sentence_length = 25
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
                line = line.strip().replace(',', ' ')
                j = 0
                print("before handle_hashtags: {}".format(line))
                line = handle_hashtags(line, vocab)
                print("after handle_hashtags: {}".format(line))



def prepare_data(train_pos_file, train_neg_file, train_size, vocab, max_sentence_length):
    """
    prepare training data
    """
    train_X = np.zeros((train_size, max_sentence_length))
    train_Y = np.zeros((train_size, 2))
    # sanity check because we initialize with zero then we don't have to do padding
    assert vocab['<PAD/>'] == 0
    i = 0
    pos = 0
    cut = 0
    empty = 0
    for filename in [train_neg_file, train_pos_file]:
        with open(filename) as f:
            for line in f:
                line = line.strip().replace(',', ' ')
                j = 0
                #print("before handle_hashtags: {}".format(line))
                line = handle_hashtags(line, vocab)
                #print("after handle_hashtags: {}".format(line))
                for word in line.split():
                    word = filter_with_voc(word, vocab)
                    if word is not None:
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
    return train_X, train_Y


def prepare_valid_data(max_sentence_length, vocab):
    validate_x = np.zeros((VALID_SIZE, max_sentence_length))
    i = 0
    cut = 0
    empty = 0
    with open(VALID_FILE_NAME) as f:
        for line in f:
            line = line.strip().split(',')
            tweet = ' '.join(line[1:])
            tweet = tweet.strip().replace(',', ' ')
            tweet = handle_hashtags(tweet, vocab)
            j = 0
            for word in tweet.split():
                filtered_word = filter_with_voc(word, vocab)
                if filtered_word is not None:
                    validate_x[i, j] = vocab[filtered_word]
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
    print("usage: preprocess.py  [--full] [--sentence-length=]")
    print("\t--full use full data set")
    print("\t--sentence-length= maximum sentence length")


def main(argv):
    max_sentence_length = 30
    train_pos_file = POS_FILE_NAME
    train_neg_file = NEG_FILE_NAME
    train_size = SMALL_TRAIN_SIZE
    try:
        opts, args = getopt.getopt(argv, "[fl:]", ["full", "sentence-length="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--full"):
            train_pos_file = FULL_POS_FILE_NAME
            train_neg_file = FULL_NEG_FILE_NAME
            train_size = FULL_TRAIN_SIZE
        if opt in ("-s", "--sentence-length"):
            max_sentence_length = int(arg)

    print('Pickle vocabulary..')
    vocab = pickle_vocab()

    print('Pickle word embeddings (this can take some time)..')
    pickle_word_embeddings(vocab)

    print('prepare training data..')
    X, Y = prepare_data(train_pos_file,train_neg_file, train_size, vocab, max_sentence_length)
    np.save('../data/preprocessing/trainX', X)
    np.save('../data/preprocessing/trainY', Y)

    print('prepare validation data..')
    validate_x = prepare_valid_data(max_sentence_length, vocab)
    np.save('../data/preprocessing/validateX', validate_x)


if __name__ == "__main__":
    main(sys.argv[1:])
    #test_handle_hashtags()
