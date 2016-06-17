#!/bin/bash

echo "This script is no longer necessary."
exit 1

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
#
cat ../data/preprocessing/vocab.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 | tr ',' '\n' | sed '/^[[:space:]]*$/d' > ../data/preprocessing/vocab_cut.txt
