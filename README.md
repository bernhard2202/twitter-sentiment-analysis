# twitter-sentiment-analysis

## Overview

Twitter Sentiment Analysis with Deep Convolutional Neural Networks and
LSTMs.

Authors: Andrei Bârsan (@AndreiBarsan), Bernhard Kratzwald (@bernhard2202), Nikost Kolitsas (@NikosKolitsas).

## Setting up

 1. Ensure that the [Twitter data files from Kaggle][0] is in the `data/train` and `data/test` folders.
 2. Run the preprocessing algorithm:
    ```bash
    (cd preprocessing && ./run_preprocessing.sh)
    ```
 3. Train the LSTM pipeline on Euler:
    a) Ensure that 'euler' points to the right user and hostname
       in you sshconfig.
    b) Ensure that you have all local dependencies installed (preferably
       in a virtual environment).
    ```bash
    pip install -r requirements.txt
    ```
    c) Start the process using Fabric3.
    ```bash
    fab euler:run    
    ```
    d) Wait roughly 36 hours. Fabric3 is smart enough to tell LSF to
       email you when the job kicks off, and when it completes.
    e) Use Fabric3 to grab the results:
    ```bash
    fab euler:fetch
    ```
 4. Use one of the downloaded checkpoints to compute the prediction:
    ```bash
    python -m predict --checkpoint_file data/runs/euler/<your-run>/checkpoints/model-<step-count>
    ```
 5. To train e.g. the CNN pipeline, modify `fabfile.py` accordingly, so
    that the `--nolstm` flag is used, and then repeat the other steps.


[0]:https://inclass.kaggle.com/c/cil-text-classification/data


