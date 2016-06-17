#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
#original
#cat ../data/train/train_pos_full.txt ../data/train/train_neg_full.txt ../data/test/test_data.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ../data/preprocessing/vocab.txt

#modified
if [ ! -f ../data/train/train_pos_full_orig.txt ]; then
    echo "creating copy of train_pos_full.txt file"
    mv ../data/train/train_pos_full.txt ../data/train/train_pos_full_orig.txt
fi
if [ ! -f ../data/train/train_neg_full_orig.txt ]; then
    echo "creating copy of train_pos_full.txt file"
    mv ../data/train/train_neg_full.txt ../data/train/train_neg_full_orig.txt
fi
if [ ! -f ../data/test/test_data_orig.txt ]; then
    echo "creating copy of train_pos_full.txt file"
    mv ../data/test/test_data.txt ../data/test/test_data_orig.txt
fi


python3 preprocess_step1.py
cat ../data/train/train_pos_full.txt ../data/train/train_neg_full.txt ../data/test/test_data.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ../data/preprocessing/vocab.txt
cat ../data/preprocessing/vocab.txt | sed "s/^\s\+//g" | sort -rn > ../data/preprocessing/vocab_cut.txt


