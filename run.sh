#!/usr/bin/env bash

echo "Will now grab checkpoints, and perform our team's top prediction."
echo "Make sure you read the README and install all Python dependencies before running this!"

mkdir -p data

if ! [[ -d data/key-checkpoints ]]; then
    if ! [[ -f data/key-checkpoints.zip ]]; then
        echo "You don't have the zip. Downloading from Polybox..."
        wget https://polybox.ethz.ch/index.php/s/t3rfSesq0fGuBW1/download \
             -O data/key-checkpoints.zip
    fi

    echo "You have the zip but not the folder, extracting..."
    (cd data && unzip key-checkpoints.zip)
fi

echo "Performing predictions..."

python -m ensemble --notrain_error \
    --checkpoint_file           data/key-checkpoints/best-lstm-68000 \
    --second_checkpoint_file    data/key-checkpoints/best-cnn-68000

