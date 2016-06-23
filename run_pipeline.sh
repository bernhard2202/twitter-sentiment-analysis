#!/usr/bin/env bash

cd ./preprocessing/
./build_vocab.sh
./preprocess.py --full True --vocab_has_counts True --advanced True --pretrained_w2v True
cd ..
python3 ./run_pipeline.sh --test_split 500
