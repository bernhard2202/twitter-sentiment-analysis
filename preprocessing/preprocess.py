#!/usr/bin/env python3
import pickle
import gensim
import numpy as np
# from stop_words import get_stop_words
import re


# stopwords = set(get_stop_words('en'))
maxSentenceLength = 25
trainSize = 2500000
# numberPattern = re.compile('^[0-9]+.[0-9]*$')


def word_filter(word):
    word = ''.join(i for i in word if not i.isdigit())
    if len(word) < 2:
        return None
    # remove hashtags in beginning
    if word[0] == '#':
        word = word[1:]

    # stopwords
    # if word in stopwords:
    #    return None

    # remove numbers
    # if numberPattern.match(word) is not None:
    #    return None

    # remove single chars
    if len(word) < 2:
        return None

    # todo add hastag splitting maybe?
    return word


def filter_with_voc(word, voc):
    word = word_filter(word)
    if (word is not None) and (word in voc):
        return word
    else:
        return None


def main():
    print('Pickle vocabulary..')

    '''
    pickle vocabulary
    '''

    vocab = dict()
    vocabinv = dict()
    # pre insert the padding word
    index = 0
    vocab["<PAD/>"] = index
    vocabinv[index] = "<PAD/>"
    index += 1
    i = 1
    words = set()

    with open('../data/preprocessing/vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            word = word_filter(line.strip())
            # word not filtered in preprocessing and word
            # unique after filtering
            i += 1
            if (word is not None) and (word not in words):
                words.add(word)
                vocab[word] = index
                vocabinv[index] = word
                index += 1

    # sanity check
    assert len(vocab) == (len(words) + 1) == len(vocabinv)

    with open('../data/preprocessing/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, protocol=2)
    with open('../data/preprocessing/vocab-inv.pkl', 'wb') as f:
        pickle.dump(vocabinv, f, protocol=2)

    # release some memory
    vocabinv = words = None

    print("Vocabulary pickled.")
    print("Total number of unique words = {}; words filterd by preprocessing = {}".format(len(vocab), (i - index)))

    '''
    pickle word embeddings
    '''

    print("Loading word2vec embeddings.. (this can take some time)")
    model = gensim.models.word2vec.Word2Vec.load_word2vec_format('../data/word2vec/GoogleNews-vectors-negative300.bin',
                                                                 binary=True)
    # todo more elegant way
    embedding_dim = len(model['queen'])
    changed = 0
    X = np.random.uniform(-0.25, 0.25, size=(len(vocab), embedding_dim))
    print("create word2vec lookup table..")
    for word in vocab:
        if word in model:
            changed += 1
            X[vocab[word]] = model[word]
    model = []
    np.save('../data/preprocessing/embeddings', X)
    print("Embeddings pickeled.")
    print("Used {} pre-trained word2vec vectors and {} new random vectors.".format(changed, (len(vocab) - changed)))

    '''
    prepare training data
    '''

    print('prepare training data..')
    train_X = np.zeros((trainSize, maxSentenceLength))
    train_Y = np.zeros((trainSize, 2))
    # sanity check because we initialize with zero then we dont have to do padding
    assert vocab['<PAD/>'] == 0
    i = 0
    pos = 0
    cut = 0
    empty = 0
    for filename in ['../data/train/train_pos_full.txt', '../data/train/train_neg_full.txt']:
        with open(filename) as f:
            for line in f:
                line = line.strip().replace(',', ' ')
                j = 0
                for word in line.split():
                    word = filter_with_voc(word, vocab)
                    if word is not None:
                        train_X[i, j] = vocab[word]
                        j += 1
                    if j == maxSentenceLength:
                        cut += 1
                        # print("cut: "+line)
                        # cut sentences longer than max sentence lenght
                        break
                if j == 0:
                    empty += 1
                    # print("empty: "+line)
                if filename == '../data/train/train_pos_full.txt':
                    train_Y[i, 0] = 0
                    train_Y[i, 1] = 1
                    pos += 1
                else:
                    train_Y[i, 1] = 0
                    train_Y[i, 0] = 1

                i += 1
    assert pos == (len(train_Y) / 2)
    assert train_Y.shape[0] == train_X.shape[0] == i
    np.save('../data/preprocessing/trainX', train_X)
    np.save('../data/preprocessing/trainY', train_Y)
    print("Preprocessing done. {} tweets cut to max sentence lenght and {} tweets disapeared due to filtering."
          .format(cut, empty))

    '''
    preprocess validation data
    '''
    print('Proprocessing the validation set..')
    validate_x = np.zeros((10000, maxSentenceLength))
    i = 0
    cut = 0
    empty = 0
    with open('../data/test/test_data.txt') as f:
        for line in f:
            line = line.strip().split(',')
            tweet = ' '.join(line[1:])
            tweet = tweet.strip().replace(',', ' ')
            j = 0
            for word in tweet.split():
                filtered_word = filter_with_voc(word, vocab)
                if filtered_word is not None:
                    validate_x[i, j] = vocab[filtered_word]
                    j += 1
                if j == maxSentenceLength:
                    cut += 1
                    # print("cut: "+line)
                    # cut sentences longer than max sentence lenght
                    break
            if j == 0:
                print(tweet)
                empty += 1
            i += 1
    print("Preprocessing done. {} tweets cut to max sentence lenght and {} tweets disapeared due to filtering."
          .format(cut, empty))
    np.save('../data/preprocessing/validateX', validate_x)


if __name__ == '__main__':
    main()
