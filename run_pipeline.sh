#!/usr/bin/env bash

./preprocessing/build_vocab.sh
./preprocessing/preprocess.py --full True --vocab_has_counts True --advanced True --pretrained_w2v True
python3 ./run_pipeline.sh --test_split 500
