#!/usr/bin/env bash

./build_vocab.sh
./preprocess.py --full True --vocab_has_counts True --advanced True --pretrained_w2v True
