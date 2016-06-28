#!/usr/bin/env bash

# Data files (train and test).
posFile="../data/train/train_pos_full.txt"
negFile="../data/train/train_neg_full.txt"
testFile="../data/test/test_data.txt"

# Backups of the original data files.
posFileBak="../data/train/train_pos_full_orig.txt"
negFileBak="../data/train/train_neg_full_orig.txt"
testFileBak="../data/test/test_data_orig.txt"

# Output files.
vocabFile="../data/preprocessing/vocab.txt"
cutVocabFile="../data/preprocessing/vocab_cut.txt"

#modified
if [ ! -f "$posFileBak" ]; then
    echo "Creating copy of positive training file [$posFile] as [$posFileBak]."
    cp "$posFile" "$posFileBak"
    #mv ../data/train/train_pos_full.txt ../data/train/train_pos_full_orig.txt
fi
if [ ! -f "$negFileBak" ]; then
    echo "Creating copy of negative training file [$negFile] as [$negFileBak]."
    cp "$negFile" "$negFileBak"
    #mv ../data/train/train_neg_full.txt ../data/train/train_neg_full_orig.txt
fi
if [ ! -f "$testFileBak" ]; then
    echo "Creating copy of test file [$testFile] as [$testFileBak]."
    cp "$testFile" "$testFileBak"
    #mv ../data/test/test_data.txt ../data/test/test_data_orig.txt
fi

# Run the first stage of preprocessing right away: this does some smart
# substitutions, like replacing numbers with '<num>' tokens.
#echo 'NOT doing preprocessing step #1!!!'
python3 pattern_matching.py

echo 'Finished preprocessing step #1.'

echo 'Building vocabulary...'
cat "$posFile" "$negFile" "$testFile" | sed "s/ /\n/g" |
  grep -v "^\s*$" | sort | uniq -c > "$vocabFile"

# Do the cutting right away (no need for second script).
# Note that this no longer strips away rare words!
echo 'Cutting vocabulary into tokens... (NOT removing rare words).'
cat "$vocabFile" | sed "s/^\s\+//g" | sort -rn > "$cutVocabFile"
#cat ../data/preprocessing/vocab.txt | sed "s/^\s\+//g" | sort -rn > ../data/preprocessing/vocab_cut.txt

echo 'Generate word mappings'
python3 ./word_mappings.py

echo 'Finished vocabulary processing.'

# Old version of the file.
# TODO(bernhard): remove once preprocessing is fixed.
#
# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
#
#cat "$posFile" "$negFile" "$testFile" | sed "s/ /\n/g" |
#    grep -v "^\s*$" | sort | uniq -c > "$vocabFile"
