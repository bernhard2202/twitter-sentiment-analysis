#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ../data/train/train_pos_full.txt ../data/train/train_neg_full.txt ../data/test/test_data.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ../data/preprocessing/vocab.txt
