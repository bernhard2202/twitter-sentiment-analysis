""" local word2vec embedding training """
import os
from gensim.models.word2vec import Word2Vec

DATA_PATH = '../data'
TRAIN = os.path.join(DATA_PATH, 'train')
TEST = os.path.join(DATA_PATH, 'test')
POS_TWEET_FILE = os.path.join(TRAIN, 'train_pos_full_orig.txt')
NEG_TWEET_FILE = os.path.join(TRAIN, 'train_neg_full_orig.txt')
TEST_TWEET_FILE = os.path.join(TEST, 'test_data_orig.txt')
EMBEDDING_SIZE = 20

def read_tweets(fname):
    """Read the tweets in the given file.

    Returns a 2d array where every row is a tweet, split into words.
    """
    with open(fname, 'r') as f:
        return [l.split() for l in f.readlines()]

pos_tweets = read_tweets(POS_TWEET_FILE)
neg_tweets = read_tweets(NEG_TWEET_FILE)
test_tweets = read_tweets(TEST_TWEET_FILE)
sentences = pos_tweets + neg_tweets + test_tweets
print(len(sentences))

tokens = [item.strip() for sentence in sentences for item in sentence]

WORKERS = 8
model = Word2Vec(sentences, size=EMBEDDING_SIZE, window=10, min_count=5, workers=WORKERS)

fname = "{0}/word2vec/word2vec-local-gensim-orig-{1}.bin".format(DATA_PATH, EMBEDDING_SIZE)
print("Writing embeddings to file {0}.".format(fname))
model.save(fname)
print("Done! Happy neural networking!")