#!/usr/bin/env python3

import pickle

from gensim.models.word2vec import Word2Vec
import numpy as np
# Use the Google flags library distributed with TensorFlow.
import tensorflow as tf

# TODO(andrei): Convert all these to gflags as well.
FULL_POS_FILE_NAME = "../data/train/train_pos_full.txt"
FULL_NEG_FILE_NAME = "../data/train/train_neg_full.txt"
POS_FILE_NAME = "../data/train/train_pos.txt"
NEG_FILE_NAME = "../data/train/train_neg.txt"
VALID_FILE_NAME = "../data/test/test_data.txt"
VOCAB_FILE_NAME = "../data/preprocessing/vocab_cut.txt"

tf.flags.DEFINE_boolean("pretrained_w2v", False, "Whether to use official Google pre-trained"
                                                 " word2vec vectors, or locally-trained ones."
                                                 " (default: False)")
tf.flags.DEFINE_string("pretrained_w2v_file",
                       "../data/word2vec/GoogleNews-vectors-negative300.bin",
                       "The name of the pre-trained word2vec embedding file."
                       " (default: ../data/word2vec/GoogleNews-vectors-negative300.bin")
tf.flags.DEFINE_string("local_w2v_file",
                       None,
                       "Name of the word2vec embedding file trained locally on"
                       " the Twitter dataset.")
tf.flags.DEFINE_boolean("advanced", False, "Whether to use advanced"
                                           " preprocessing (e.g. spelling"
                                           " correction). (default: False)")
tf.flags.DEFINE_boolean("full", False, "Whether to use the full Twitter dataset."
                                       " (default: False)")
tf.flags.DEFINE_integer("sentence_length", 30, "The maximum sentence length to"
                                               " consider (in words)."
                                               " (default: 30)")
# TODO(andrei): Unify this part with Nikos's advanced second stage of corrections.
tf.flags.DEFINE_boolean("split_hashtags", True, "Whether to attempt to split hashtags.")
tf.flags.DEFINE_boolean("vocab_has_counts", False,
                        "Whether the cut vocabulary file given also contains"
                        " each token's counts. This is the case when using"
                        " Nikos's preprocessing scheme.")

# tf.flags.DEFINE_integer("min_occurrence_count", 5,
#                         "The minimum number of times a word has to appear in"
#                         " our corpus in order to consider a real word vector"
#                         " for it, as opposed to just giving it a random one."
#                         " This flag is useful when using Nikos's preprocessing"
#                         " scheme, since this scheme no longer pre-trims"
#                         " the words by their count in 'vocab_cut.txt'.")

FLAGS = tf.flags.FLAGS

# TODO(andrei): set these automatically from the input files.
VALID_SIZE = 10000
FULL_TRAIN_SIZE = 2500000
SMALL_TRAIN_SIZE = 200000

# TODO(andrei): Option to ignore mappings.
# TODO(andrei): Clean-up mapping loading code.
MAPPINGS_FOLDER = "../data/preprocessing/mappings/"
print("Loading pre-computed mappings...")
with open(MAPPINGS_FOLDER + "mappings.pkl", 'rb') as f:
    (mappings, pretrained, extra_words) = pickle.load(f)
print("Finished loading pre-computed mappings.")


def word_filter(word):
    # TODO(andrei): Is this still necessary after Nikos's stage 1 preprocessing?
    word = ''.join(i for i in word if not i.isdigit())
    if len(word) < 2:
        return None

    # remove hashtags in beginning
    # if word[0] == '#':
    #     return None
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


def pickle_vocab_and_embeddings(file_prefix, has_counts):
    """
    pickle vocabulary

    Arguments:
        'has_counts': Whether the cut vocabulary file given also contains each
                      token's counts. This is the case when using Nikos's
                      preprocessing scheme.
    """
    vocab = dict()
    vocab_inv = dict()
    # pre insert the padding word
    index = 0
    vocab["<PAD/>"] = index
    vocab_inv[index] = "<PAD/>"
    index += 1
    # i = 1
    # words = set()

    print("WARNING: IGNORING 'has_counts' flag for now while testing custom mappings.")
    if has_counts:
        print("Assuming vocabulary file also contains frequencies.")
    else:
        print("Assuming vocabulary file contains no frequency info and just has"
              " exactly one token per line.")

    for word in extra_words:
        if word in pretrained:
            print("In extra_words and pretrained simultaneously!: " + word)

        vocab[word] = index
        vocab_inv[index] = word
        index += 1

    # Handle word embeddings

    if FLAGS.pretrained_w2v:
        fname = FLAGS.pretrained_w2v_file
        print("Using pre-trained word2vec vectors from '{0}'.".format(fname))
        model = Word2Vec.load_word2vec_format(fname, binary=True)
    else:
        fname = FLAGS.local_w2v_file
        print("Using locally-trained word2vec vectors from '{0}'.".format(
            fname))
        model = Word2Vec.load(fname)

    embedding_dim = model.vector_size
    X = np.empty( (len(extra_words)+len(pretrained)+1 ,embedding_dim) )
    X[0:len(extra_words)+1] = np.random.uniform(-0.25, 0.25,
                                                size=(len(extra_words)+1, embedding_dim))
    print("create word2vec lookup table..")

    assert index == len(extra_words)+1

    # TODO(andrei): Pass 'pretrained' to this function.
    for word in pretrained:
        vocab[word] = index
        vocab_inv[index] = word
        X[index] = model[word]
        index += 1

    # This was the old functionality.
    # with open(VOCAB_FILE_NAME) as f:
    #     for idx, line in enumerate(f):
    #         if has_counts:
    #             # Line has format '<count> <token>'.
    #             freq, word = line.split()
    #             word = word_filter(word.strip())
    #
    #             # In this scenario, 'vocab_cut' still includes very rare words,
    #             # so we need to ensure we don't save their embeddings, since
    #             # they're probably useless and just take up disk space.
    #             if int(freq) < FLAGS.min_occurrence_count:
    #                 continue
    #         else:
    #             # Line has format '<token>'.
    #             word = word_filter(line.strip())
    #
    #         # word not filtered in preprocessing and word
    #         # unique after filtering
    #         i += 1
    #         if (word is not None) and (word not in words):
    #             words.add(word)
    #             vocab[word] = index
    #             vocab_inv[index] = word
    #             index += 1

    # sanity checks
    # assert len(vocab) == (len(words) + 1) == len(vocab_inv)
    print("len(vocab)= {}".format(len(vocab)))
    print("len(vocab_inv)= {}".format(len(vocab_inv)))
    print("len(extra_words)+len(pretrained)+1= {}".format(len(extra_words)+len(pretrained)+1))

    # Save vocabulary
    vocab_fname = '../data/preprocessing/{}-vocab.pkl'.format(file_prefix)
    vocab_inv_fname = '../data/preprocessing/{}-vocab-inv.pkl'.format(file_prefix)
    with open(vocab_fname, 'wb') as f:
        pickle.dump(vocab, f, protocol=2)
    with open(vocab_inv_fname, 'wb') as f:
        pickle.dump(vocab_inv, f, protocol=2)

    print("Vocabulary pickled in [{}] and [{}].".format(vocab_fname, vocab_inv_fname))
    # print("Total number of unique words = {}; words filterd by preprocessing = {}".format(len(vocab), (i - index)))

    # Save embeddings
    np.save('../data/preprocessing/{}-embeddings'.format(file_prefix), X)
    print("Embeddings pickled.")
    print("Used {} pre-trained word2vec vectors and {} new random vectors."
          .format(len(pretrained), len(extra_words)+1))

    return vocab


def pickle_word_embeddings(vocab, file_prefix):
    """Pickles word embeddings into a easy-to-load numpy format.

    Only pickles embeddings which we actually use, which is especially useful
    when dealing with huge pre-trained embedding datasets.
    """
    raise RuntimeError("Deprecated.")
    # TODO(andrei): Support hybrid approach.
    if FLAGS.pretrained_w2v:
        fname = FLAGS.pretrained_w2v_file
        print("Using pre-trained word2vec vectors from '{0}'.".format(fname))
        model = Word2Vec.load_word2vec_format(fname, binary=True)
    else:
        fname = FLAGS.local_w2v_file
        print("Using locally-trained word2vec vectors from '{0}'.".format(
            fname))
        model = Word2Vec.load(fname)

    embedding_dim = model.vector_size
    changed = 0

    # TODO(andrei): Consider initializing unknown embeddings with smaller
    # values.
    X = np.random.uniform(-0.25, 0.25, size=(len(vocab), embedding_dim))
    print("Creating word2vec lookup table...")
    print("Model stats:")
    print("Corpus size:                     {0}".format(model.corpus_count))
    print("Vector dimensionality:           {0}".format(embedding_dim))
    print("Bash-generated vocabulary size:  {0}".format(len(vocab)))
    print("W2V model vocabulary size:       {0}".format(len(model.vocab.keys())))

    print("Sample from w2v model: ", list(model.vocab.keys())[:25])
    print("Sample from vocabulary:", list(list(vocab.keys())[:25]))

    for word in vocab:
        if word in model:
            changed += 1
            X[vocab[word]] = model[word]

    np.save('../data/preprocessing/{}-embeddings'.format(file_prefix), X)
    print("Embeddings pickled.")
    print("Used {} pre-trained word2vec vectors and {} new random vectors."
          .format(changed, (len(vocab) - changed)))


def handle_hashtags_and_mappings(line, vocab):
    # Part of Nikos's second stage of preprocessing.
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
                        if ~np.any(claimed[s:s+n]):
                            # nothing is claimed so take it
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


def handle_hashtags(line, vocab):
    result_line = []
    for word in line.split():
        if word[0] == '#':
            # TODO(andrei): Porque no los dos? Option to keep both split words
            # and original hashtag.
            if not FLAGS.split_hashtags:
                # Just return the full original hashtag if we don't want to
                # split it.
                result_line.append(word)
                continue

            word = word[1:]
            length = len(word)
            word_result = []

            # initially all letters are free to select
            claimed = np.full(length, False, dtype=bool)
            # initially search for words with n letters, then n-1,... until 1
            # letter words
            for n in range(length, 0, -1):
                # starting point. so we examine substring  [s,s+n)
                for s in range(0, length-n+1):
                    substring = word[s:s+n]
                    if substring in vocab:
                        if ~np.any(claimed[s:s+n]):
                            # nothing is claimed so take it
                            claimed[s:s+n] = True
                            word_result.append((s, substring))
            word_result.sort()
            for _, substring in word_result:
                result_line.append(substring)
        else:
            result_line.append(word)
    return ' '.join(result_line)


def prepare_data(train_pos_file, train_neg_file, train_size, vocab,
                 max_sentence_length):
    """Prepares the training data."""
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
                # line = line.strip().replace(',', ' ')
                line = line.strip()
                j = 0
                #print("before handle_hashtags: {}".format(line))
                if FLAGS.advanced:
                    line = handle_hashtags_and_mappings(line, vocab)
                else:
                    line = handle_hashtags(line, vocab)
                #print("after handle_hashtags: {}".format(line))

                if FLAGS.advanced:
                    for word in line.split():
                        if word in vocab:
                            train_X[i, j] = vocab[word]
                            j += 1
                        if j == max_sentence_length:
                            cut += 1
                            # cut sentences longer than max sentence length
                            break
                else:
                    for word in line.split():
                        word = filter_with_voc(word, vocab)
                        if word is not None:
                            train_X[i, j] = vocab[word]
                            j += 1
                        if j == max_sentence_length:
                            cut += 1
                            # cut sentences longer than max sentence length
                            break

                if j == 0:
                    empty += 1

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

    print("{} tweets cut to max sentence length and {} tweets disappeared due"
          " to filtering.".format(cut, empty))
    return train_X, train_Y


def prepare_valid_data_old(max_sentence_length, vocab):
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
                    # cut sentences longer than max sentence length
                    break
            if j == 0:
                #print(tweet)
                empty += 1
            i += 1
    print("Preprocessing done. {} tweets cut to max sentence length and {} tweets disapeared due to filtering."
          .format(cut, empty))
    return validate_x


def prepare_valid_data(max_sentence_length, vocab):
    assert FLAGS.advanced, "Should only be used in advanced mode."
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


def main():
    max_sentence_length = FLAGS.sentence_length

    if FLAGS.full:
        train_pos_file = FULL_POS_FILE_NAME
        train_neg_file = FULL_NEG_FILE_NAME
        train_size = FULL_TRAIN_SIZE
        prefix = "full"
        print("Using full vocabulary, and ALL training tweets.")
    else:
        train_pos_file = POS_FILE_NAME
        train_neg_file = NEG_FILE_NAME
        train_size = SMALL_TRAIN_SIZE
        prefix = "subset"
        print("Using full vocabulary, but only subset of tweets.")

    if FLAGS.advanced:
        print("Using ADVANCED preprocessing (e.g. automatic spelling "
              " correction, smart hashtag splitting, etc.).")
        if FLAGS.split_hashtags:
            print("Ignoring 'split_hashtags' flag in this mode, and doing"
                  " smart splitting anyway.")
        if not FLAGS.pretrained_w2v:
            print("WARNING: You are using locally-trained word2vec embeddings"
                  " with ADVANCED preprocessing. This subsystem was designed"
                  " with pretrained embeddings in mind!")
    else:
        print("Using SIMPLE preprocessing (e.g. NO spelling correction).")
        if FLAGS.split_hashtags:
            # TODO(andrei): Ensure message correctness.
            print("Will attempt to split encountered hashtags.")
        else:
            print("Will NOT attempt to split encountered hashtags.")

    print("Pickling vocabulary and embeddings (this can take some time)...")
    vocab = pickle_vocab_and_embeddings(prefix, FLAGS.vocab_has_counts)

    print("Preparing training data...")
    X, Y = prepare_data(train_pos_file,train_neg_file, train_size, vocab,
                        max_sentence_length)
    np.save("../data/preprocessing/{0}-trainX".format(prefix), X)
    np.save("../data/preprocessing/{0}-trainY".format(prefix), Y)

    print('Preparing validation data...')
    if FLAGS.advanced:
        validate_x = prepare_valid_data(max_sentence_length, vocab)
    else:
        validate_x = prepare_valid_data_old(max_sentence_length, vocab)
    np.save("../data/preprocessing/validateX", validate_x)


if __name__ == "__main__":
    main()
