#!/usr/bin/env python3

import os
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
# Mappings used by the advanced preprocessing code, computed by 'word_mappings.py'.
MAPPINGS_FOLDER = "../data/preprocessing/mappings/"

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
tf.flags.DEFINE_boolean("split_hashtags", True, "Whether to attempt to split hashtags.")
tf.flags.DEFINE_boolean("vocab_has_counts", False,
                        "Whether the cut vocabulary file given also contains"
                        " each token's counts. This is the case when using"
                        " Nikos's preprocessing scheme.")
tf.flags.DEFINE_integer("min_occurrence_count", 5,
                        "The minimum number of times a word has to appear in"
                        " our corpus in order to consider a real word vector"
                        " for it, as opposed to just giving it a random one."
                        " This flag is useful when using Nikos's preprocessing"
                        " scheme, since this scheme no longer pre-trims"
                        " the words by their count in 'vocab_cut.txt'.")

FLAGS = tf.flags.FLAGS

# TODO(andrei): set these automatically from the input files.
VALID_SIZE = 10000
FULL_TRAIN_SIZE = 2500000
SMALL_TRAIN_SIZE = 200000


def load_mappings():
    mappings_fname = os.path.join(MAPPINGS_FOLDER, "mappings.pkl")
    with open(mappings_fname, 'rb') as f:
        (mappings, pretrained, extra_words) = pickle.load(f)
    print("Finished loading pre-computed mappings.")
    return mappings, pretrained, extra_words


def word_filter(word):
    word = ''.join(i for i in word if not i.isdigit())
    if len(word) < 2:
        return None
    else:
        return word


def filter_with_voc(word, voc):
    word = word_filter(word)
    if word is not None and (word in voc):
        return word
    else:
        return None


def pickle_vocab_and_embeddings(file_prefix, has_counts, pretrained,
                                extra_words):
    """
    Pickles the vocabulary and embeddings for easy loading in training stage.

    Arguments:
        file_prefix: Prefix to use when saving output, useful for
                     discerning between saved data for full or partial (subset)
                     training.
        has_counts: Whether the cut vocabulary file given also contains each
                    token's counts. This is the case when using Nikos's
                    preprocessing scheme.
        pretrained: A map of 'word' -> 'embedding' for pretrained word vectors.
                    Only used when '--advanced' is enabled.
        extra_words: A list of words which don't have their own pre-trained
                     embedding, but are still common enough to want to keep
                     around.
                     Only used when '--advanced' is enabled.
    """
    vocab = dict()
    vocab_inv = dict()
    # pre insert the padding word
    index = 0
    vocab["<PAD/>"] = index
    vocab_inv[index] = "<PAD/>"
    index += 1

    if has_counts:
        print("Assuming vocabulary file also contains frequencies.")
    else:
        print("Assuming vocabulary file contains no frequency info and just has"
              " exactly one token per line.")

    if FLAGS.advanced:
        print("Processing extra words.")
        print("We have {0} extra words.".format(len(extra_words)))
        for word in extra_words:
            if word in pretrained:
                print("WARNING! In extra_words and pretrained simultaneously!: " + word)

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

    # Print some detailed information about the embeddings.
    embedding_dim = model.vector_size
    print("Creating word2vec lookup table...")
    print("Model stats:")
    if hasattr(model, 'corpus_count'):
        print("Corpus size:                     {0}".format(model.corpus_count))
    print("Vector dimensionality:           {0}".format(embedding_dim))
    print("Bash-generated vocabulary size:  {0}".format(len(vocab)))
    print("W2V model vocabulary size:       {0}".format(len(model.vocab.keys())))

    print("Sample from w2v model: ", list(model.vocab.keys())[:25])
    print("Sample from vocabulary:", list(list(vocab.keys())[:25]))

    if FLAGS.advanced:
        assert index == len(extra_words)+1

        X = np.empty((len(extra_words)+len(pretrained)+1, embedding_dim))
        X[0:len(extra_words)+1] = np.random.uniform(-0.25, 0.25,
                                                    size=(len(extra_words)+1,
                                                          embedding_dim))
        for word in pretrained:
            vocab[word] = index
            vocab_inv[index] = word
            X[index] = model[word]
            index += 1

        print("len(vocab)= {}".format(len(vocab)))
        print("len(vocab_inv)= {}".format(len(vocab_inv)))
        print("len(extra_words)+len(pretrained)+1= {}".format(
            len(extra_words)+len(pretrained)+1))
    else:
        words = set()
        with open(VOCAB_FILE_NAME) as f:
            for idx, line in enumerate(f):
                if has_counts:
                    # Line has format '<count> <token>'.
                    freq, word = line.split()
                    word = word_filter(word.strip())

                    # In this scenario, 'vocab_cut' still includes very rare
                    # words, so we need to ensure we don't save their
                    # embeddings, since they're probably useless and would just
                    # take up disk space.
                    if int(freq) < FLAGS.min_occurrence_count:
                        continue
                else:
                    # Line has format '<token>'.
                    word = word_filter(line.strip())

                # word not filtered in preprocessing and word
                # unique after filtering
                if (word is not None) and (word not in words):
                    words.add(word)
                    vocab[word] = index
                    vocab_inv[index] = word
                    index += 1

        # Sanity check
        assert len(vocab) == (len(words) + 1) == len(vocab_inv)

        # Now we finally set X up.
        X = np.random.uniform(-0.25, 0.25, size=(len(vocab), embedding_dim))
        changed = 0
        for word in vocab:
            if word in model:
                changed += 1
                X[vocab[word]] = model[word]

    # Save vocabulary
    vocab_fname = '../data/preprocessing/{}-vocab.pkl'.format(file_prefix)
    vocab_inv_fname = '../data/preprocessing/{}-vocab-inv.pkl'.format(file_prefix)
    with open(vocab_fname, 'wb') as f:
        pickle.dump(vocab, f, protocol=2)
    with open(vocab_inv_fname, 'wb') as f:
        pickle.dump(vocab_inv, f, protocol=2)

    print("Vocabulary pickled in [{}] and [{}].".format(vocab_fname, vocab_inv_fname))

    # Save embeddings
    np.save('../data/preprocessing/{}-embeddings'.format(file_prefix), X)
    print("Embeddings pickled.")

    if FLAGS.advanced:
        print("Used {} pre-trained word2vec vectors and {} new random vectors."
              .format(len(pretrained), len(extra_words)+1))
    else:
        print("Used {} pre-trained word2vec vectors and {} new random vectors."
              .format(changed, (len(vocab) - changed)))

    return vocab


def handle_hashtags_and_mappings(line, vocab, mappings):
    """Perform hashtag processing and corrected word substitutions.

    Args:
        line: The current text line (tweet) to process.
        vocab: A map of words to their indexes.
        mappings: Word mappings, such as corrections (map wrong word to
                  corrected  one).
                  Only used when '--advanced' is enabled.

    Returns:
        The updated text line as a full string.
    """
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
                 max_sentence_length, mappings):
    """Prepares the training data.

    Args:
        mappings: See 'handle_hashtags_and_mappings'.
    """
    train_X = np.zeros((train_size, max_sentence_length))
    train_Y = np.zeros((train_size, 2))
    # sanity check because we initialize with zero then we don't have to do padding
    assert vocab['<PAD/>'] == 0
    i = 0
    pos = 0
    cut = 0
    empty = 0
    print("prepare_data: len(vocab) = {0}".format(len(vocab)))
    print("Train neg file: {0}; Train pos file: {1}".format(train_neg_file,
                                                            train_pos_file))
    for filename in [train_neg_file, train_pos_file]:
        with open(filename) as f:
            for line in f:
                # line = line.strip().replace(',', ' ')
                line = line.strip()
                j = 0
                #print("before handle_hashtags: {}".format(line))
                if FLAGS.advanced:
                    line = handle_hashtags_and_mappings(line, vocab, mappings)
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
    print("Train X shape: {0}".format(train_X.shape))
    print("Train y shape: {0}".format(train_Y.shape))
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


def prepare_valid_data(max_sentence_length, vocab, mappings):
    """
    Args:
        mappings: See 'handle_hashtags_and_mappings'.
    """
    assert FLAGS.advanced, "Should only be used in advanced mode."
    validate_x = np.zeros((VALID_SIZE, max_sentence_length))
    i = 0
    cut = 0
    empty = 0
    with open(VALID_FILE_NAME) as f:
        for tweet in f:
            tweet = tweet.strip()
            tweet = tweet[6:]   # remove prefix   "<num>,"
            tweet = handle_hashtags_and_mappings(tweet, vocab, mappings)

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
        print("\nUsing ADVANCED preprocessing (e.g. automatic spelling "
              " correction, smart hashtag splitting, etc.).\n")
        if not FLAGS.split_hashtags:
            print("Ignoring 'split_hashtags' flag in this mode, and doing"
                  " smart splitting anyway.")
        if not FLAGS.pretrained_w2v:
            print("WARNING: You are using locally-trained word2vec embeddings"
                  " with ADVANCED preprocessing. This subsystem was designed"
                  " with pretrained embeddings in mind!")
    else:
        print("Using SIMPLE preprocessing (e.g. NO spelling correction).")
        if FLAGS.split_hashtags:
            print("Will attempt to split encountered hashtags.")
        else:
            print("Will NOT attempt to split encountered hashtags.")

    if FLAGS.advanced:
        print("Loading advanced mode mappings...")
        mappings, pretrained, extra_words = load_mappings()
    else:
        mappings = pretrained = extra_words = None

    print("Pickling vocabulary and embeddings (this can take some time)...")
    vocab = pickle_vocab_and_embeddings(prefix, FLAGS.vocab_has_counts,
                                        pretrained, extra_words)

    print("Preparing training data...")
    X, Y = prepare_data(train_pos_file,train_neg_file, train_size, vocab,
                        max_sentence_length, mappings)
    np.save("../data/preprocessing/{0}-trainX".format(prefix), X)
    np.save("../data/preprocessing/{0}-trainY".format(prefix), Y)

    print('Preparing validation data...')
    if FLAGS.advanced:
        validate_x = prepare_valid_data(max_sentence_length, vocab, mappings)
    else:
        validate_x = prepare_valid_data_old(max_sentence_length, vocab)
    np.save("../data/preprocessing/validateX", validate_x)


if __name__ == "__main__":
    main()
