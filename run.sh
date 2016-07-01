#!/usr/bin/env bash

echo "Will now grab checkpoints, and perform our team's top prediction."
echo "Make sure you read the README and install all Python dependencies before running this!"

(mkdir -p data && cd data && wget ... && unzip key-checkpoints.zip)

echo "Successfully grabbed data. Performing predictions..."

python -m ensemble --notrain_error \
    --checkpoint_file           data/key-checkpoints/best-lstm-68000 \
    --second_checkpoint_file    data/key-checkpoints/best-cnn-68000

