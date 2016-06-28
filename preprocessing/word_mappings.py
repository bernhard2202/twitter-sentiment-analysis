#!/usr/bin/env python3

import os
import pickle
import re
import sys

import gensim

"""
read the sorted cut_vocab.txt     which has format:
freq word
and try to map it to word2vec pretrained words.
"""

VOCAB_FILE_NAME = "../data/preprocessing/vocab_cut.txt"

# TODO(andrei): Experiment with locally-trained embeddings.
WORD2VEC_FILE_NAME = "../data/word2vec/GoogleNews-vectors-negative300.bin"
MAPPINGS_FILE_NAME = "../data/preprocessing/mappings/mappings.pkl"
MAPPINGS_FOLDER = "../data/preprocessing/mappings/"

# Original values:
HIGH_FREQUENCY = 10
LOW_FREQUENCY = 4    # TODO experiment with different values. i believe we should increase it
HASH_TAG_FREQ_BOUND = 80
MIN_SPELL_CORRECTION_LENGTH = 5

# Optimal values on Kaggle, based on Nikos's experiments.
# HIGH_FREQUENCY = 20
# LOW_FREQUENCY = 4    # TODO experiment with different values. i believe we should increase it
# HASH_TAG_FREQ_BOUND = 10
# MIN_SPELL_CORRECTION_LENGTH = 5

if not os.path.exists(MAPPINGS_FOLDER):
    os.makedirs(MAPPINGS_FOLDER)

model = gensim.models.word2vec.Word2Vec.load_word2vec_format(WORD2VEC_FILE_NAME,
                                                             binary=True)
freq_dict = model.vocab
mappings = dict()
pretrained = set()
extra_words = dict()


# returns True if it found a correction of the word, otherwise returns False.
# The correction is with priority from word2vec. and if nothing found
# then it is from the previous inserted words in the mappings. e.g. sexxxyyys could not match to word2vec but can match with
# sexxxyyy which was previously inserted.
def spell_correction(word):
    # 1) remove 's and check if in word2vec
    # 2) remove "-" or "."  and check if in word2vec
    # 3) split "word1-word2" in "word1 word2" and check if word1 and word2 in word2vec
    # 4) look for ED=1 from word2vec
    # 5) look for ED=2 from word2vec
    # 6) look for ED=1 from previously added words   e.g. extra_words
    # 7) delete consecutive duplicate letters aaadddooraaabbblle - > adorable and then repeat 4, 5 and 6 steps
    # 8) working on the output of 7: try to split the word in two words e.g. bestmood -> best mood
    #print("spell_correction for:   "+word)
    correct_word = word
    # 1)
    if correct_word.endswith("'s"):  #remove it
        correct_word = correct_word[:-2]
        if correct_word in model:
            mappings[word] = correct_word
            pretrained.add(correct_word)
            #print("{}\t\t-->\t\t{}    just remove 's\n".format(word, mappings[word]))
            return True

    # 3)
    if '-' in correct_word or '.' in correct_word:
        #separate to distinct words
        temp = correct_word.replace("-", " ").replace(".", " ")
        temp = temp.split()     # "dcision-suport" -> "dcision suport" ->["dcision", "suport"]
        result = []
        for sub_word in temp:
            sub_word = spell_correction2(sub_word)
            if sub_word is not None:
                result.append(sub_word)
        if len(result) == len(temp):   # managed to find a correction for every single split then accept it
            mappings[word] = ' '.join(result)
            #print("split - to sep words: {}".format(mappings[word]))
            return True

    # 2) this code is executed for all words e.g. aple    and  a--p--l--e
    temp = correct_word.replace("-", "").replace(".", "")
    temp = spell_correction2(temp)
    if temp is not None:
        mappings[word] = temp
        return True

    return False


def spell_correction2(word):
    # 4) look for ED=1 from word2vec
    # 5) look for ED=2 from word2vec
    # 6) look for ED=1 from previously added words   e.g. extra_words
    # 7) delete consecutive duplicate letters aaadddooraaabbblle - > adorable and then repeat 4, 5 and 6 steps
    # 8) working on the output of 7: try to split the word in two words e.g. bestmood -> best mood
    word = word.strip()
    correct_word = None
    if word in model:
        correct_word = word   #found a solution
    # if the word is too small do not try to find a correction.
    elif len(word) >= MIN_SPELL_CORRECTION_LENGTH:
        split_words, split_score  = split_to_2_words(word)    # "decision making", 45000
        #TODO correct mistake
        ed1 = edits1(word)
        without_dupl = delete_duplicate_letters(word)
        if without_dupl != word:
            ed1.add(without_dupl)
        candidates = known(ed1)
        #print("try ED1 of word union without dupl")
        correct_word = None if len(candidates) == 0 else  max(candidates, key=vocab_sorting)
        if split_score is not None:   # take the maximum score between the best split words and the ed1 set candidates
            if correct_word is None or split_score > freq_dict.get(correct_word).count:
                correct_word = split_words
                w1, w2 = split_words.split()
                pretrained.add(w1)
                pretrained.add(w2)

        if correct_word is None and without_dupl != word and len(without_dupl) > 4:
            candidates = known(edits1(without_dupl))       #try ED=1 for the word without duplicates
            #print("try ED1 without_dupl")
            correct_word = None if len(candidates) == 0 else  max(candidates, key=vocab_sorting)

        if correct_word is None and len(word) > 4:  # try ED=1 from extra_words
            candidates = known_from_extra(edits1(word))
            correct_word = None if len(candidates) == 0 else  max(candidates, key=extra_words.get)
            #print("try ED1 from extra_words")

        #print("{}\t\t-->\t\t{}    correct1\n".format(word, correct_word) if correct_word is not None else "None")

    if correct_word is not None:   # can be from model, from extra_words, or a 2-words ("decision support" in this case
        if correct_word in model:  # we have already added the "decision" and the "support" but we are sure the "decision support" not in model)
            pretrained.add(correct_word)
        return correct_word
    return None


def delete_duplicate_letters(word):
    prev = None
    res = []
    for l in word: # for each letter check if it is the same with the previous one. if yes then omit it
        if l != prev:
            res.append(l)
        prev = l
    return ''.join(res)


def split_to_2_words(word):
    best_index = -1
    best_score = -1   # select the max
    for i in range(3, len(word)-2):
        w1, w2 = word[:i], word[i:]
        if w1 in model and w2 in model:
            cur_score = (model.vocab[w1].count + model.vocab[w2].count) / 2
            if cur_score > best_score:
                best_score = cur_score
                best_index = i

    if best_index == -1:
        return None, None
    else:
        w1, w2 = word[:best_index], word[best_index:]
        return str(w1 + " " + w2), best_score

def main():
    # maps the word that we encounter with spelled corrected word. e.g aple -> apple
    # kegkjkd ->           (discarded, if very low frequency and not similar (edit distance) with
    # something correct then do not put it in the dictionary

    with open(VOCAB_FILE_NAME) as f:
        line_cnt = 0
        for line in f:
            line_cnt += 1
            if line_cnt % 10000 == 0:
                print(line_cnt)
                #pickle_mappings()

            freq, word = line.split()
            word = word.strip()
            freq = int(freq)
            if word[0] == '#' and len(word) > 1:
                # TODO handle hashtags in special way e.g. hashtags with high frequency keep them as is and add them to vocab
                # the others are split to normal words in the preprocess_old.py
                if freq < HASH_TAG_FREQ_BOUND:  #split it to words in the next stage
                    continue
                else:
                    extra_words[word] = freq


            #try to match the word with word2vec
            if word in model:
                pretrained.add(word)
            else:
                # TODO(andrei): Keep track of more advanced statistics regarding
                # corrections and similar modifications.

                # try spelling correction
                if freq > HIGH_FREQUENCY:   # add it without doing spelling correction
                    extra_words[word] = freq
                elif spell_correction(word): # we found correction
                    pass
                else:  #correct_word==None  (nothing similar found)
                    if freq > LOW_FREQUENCY:
                        extra_words[word] = freq
                    else:
                        # very low frequency. if not manage to find a match
                        # with word2vec or previous new words just discard it
                        # TODO(andrei): Keep track of these words in case there
                        # are still patterns we could exploit.
                        pass

    pickle_mappings()


def pickle_mappings():
    print("Saving mappings to file [{0}]...".format(MAPPINGS_FILE_NAME))
    with open(MAPPINGS_FILE_NAME, "wb") as f:
        pickle.dump((mappings, pretrained, extra_words), f)

    # print them also for visual inspection
    with open(MAPPINGS_FOLDER+"mappings.txt", "w") as f:
        for x in mappings:
            f.write("{}\t\t-->\t\t{}\n".format(x, mappings[x]))

    with open(MAPPINGS_FOLDER+"extra_words.txt", "w") as f:
        for x in extra_words:
            f.write("{}\t\t-->\t\t{}\n".format(x, extra_words[x]))

    with open(MAPPINGS_FOLDER+"pretrained.txt", "w") as f:
        for x in pretrained:
            f.write(x + "\n")

    print("Finished saving mappings.")




# Modified version of Spelling Corrector from Peter Norvig.
# http://norvig.com/spell-correct.html
alphabet = 'abcdefghijklmnopqrstuvwxyz'


def edits1(word):
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in model)


def known(words): return set(w for w in words if w in model)


def correct1(word):
    candidates = known(edits1(word))
    return None if len(candidates) == 0 else  max(candidates, key=vocab_sorting)


def correct2(word):
    candidates = known_edits2(word)
    return None if len(candidates) == 0 else  max(candidates, key=vocab_sorting)


def vocab_sorting(x):
    return freq_dict.get(x).count


def known_from_extra(words): return set(w for w in words if w in extra_words)


def correct1_extra(word):   # look for ED=1 from the extra_words
    candidates = known_from_extra(edits1(word))
    return    None if len(candidates) == 0 else  max(candidates, key=extra_words.get)

if __name__ == "__main__":
    main()
