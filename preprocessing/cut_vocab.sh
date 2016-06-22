#!/bin/bash

echo "Warning: make sure you know what you're doing. 'build_vocab.sh' should" \
  " already be doing this!"

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
#
cat ../data/preprocessing/vocab.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 | tr ',' '\n' | sed '/^[[:space:]]*$/d' > ../data/preprocessing/vocab_cut.txt
