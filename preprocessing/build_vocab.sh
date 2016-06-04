#!/usr/bin/env bash

posFile="../data/train/train_pos_full.txt"
negFile="../data/train/train_neg_full.txt"
testFile="../data/test/test_data.txt"
vocabFile="../data/preprocessing/vocab.txt"

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat "$posFile" "$negFile" "$testFile" | sed "s/ /\n/g" |
    grep -v "^\s*$" | sort | uniq -c > "$vocabFile"
