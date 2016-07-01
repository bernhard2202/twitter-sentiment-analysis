# twitter-sentiment-analysis

## Overview

Twitter Sentiment Analysis with Deep Convolutional Neural Networks and LSTMs in TensorFlow.

Authors: **Andrei Bârsan** (@AndreiBarsan), **Bernhard Kratzwald** (@bernhard2202), **Nikolaos Kolitsas** (@NikosKolitsas).

Team: **Free the Varaibles!**

Computational Intelligence Lab (CIL) Project for Summer Semester 2016 at ETH Zurich.

Most of the interesting TensorFlow code is located in `train_model.py`,
`model/cnn_model.py`, and `model/lstm.py`.

## Kaggle result reproduction

In order to regenerate the top Kaggle submission, please ensure all
Python requirements are installed (next section, step 3.b) and then run:

```bash
./run.sh
```
    
This will download the top TensorFlow checkpoints from Polybox and
use them to compute the results we submitted to Kaggle. If the Polybox
file stops being available (starting in 2017), please contact
`barsana@student.ethz.ch` or follow the steps in the next section.

## Training from scratch

This project requires Python 3.5. It uses 3.5 features such as type hints.
It employs TensorFlow and scikit-learn as the main machine learning toolkits, and uses Fabric3 for launching the training pipelines remotely (e.g. to AWS or to Euler). Using a
virtual environment (e.g. `virtualenv` or Anaconda) is highly recommended.

 1. Ensure that the [Twitter data files from Kaggle][0] is in the `data/train` and `data/test` folders.
    The [pre-computed Google word2vec corups][1] must also be present in the `data/word2vec` folder.
 2. Run the preprocessing algorithm:
 
    ```bash
    (cd preprocessing && ./run_preprocessing.sh)
    ```
 3. Train the LSTM pipeline on Euler:
 
    a) Ensure that 'euler' points to the right user and hostname in you sshconfig.
    
    b) Ensure that you have all local dependencies installed (preferably in a virtual environment).
    
    ```bash
    pip install -r requirements.txt
    ```
    
    c) Start the process using Fabric3.
    
    ```bash
    fab euler:run    
    ```
    
    d) Wait roughly 36 hours. Fabric3 is smart enough to tell LSF to email you when the job kicks off, and when it completes.
    
    e) Use Fabric3 to grab the results:
    
    ```bash
    fab euler:fetch
    ```
    
 4. Use one of the downloaded checkpoints to compute the prediction:
 
    ```bash
    python -m predict --checkpoint_file data/runs/euler/<your-run>/checkpoints/model-<step-count>
    ```
    
 5. To train e.g. the CNN pipeline, modify `fabfile.py` accordingly, so that the `--nolstm` flag is used in the `_run_tf` function, and then repeat steps 3 and 4. The CNN should be faster to train (~5h over 10 epochs).
 6. One can also train things locally. For more information, run `python -m train_model --help`.


## Miscellaneous

The `ensemble.py` tool can be used to verify a trained model (checkpoint)
on the local training data to ensure that it is correct, and that the
local data isn't wrong (e.g. it hasn't been recomputed with different
preprocessing parameters, thereby making the trained model stale). This
tool can also be used to compute probability averaging from two models'
predictions by specifying a second checkpoint to load. Please run
`python -m ensemble --help` for more information.

There area also a few Jupyter notebooks in the `notebooks/` folder. Most
of them require the preprocessing to have been run first.
 * `Baselines` computes the two embedding-based baselines (averaging and concatenation).
    `preprocessing/train_word2vec.py` should be used to compute the
    local embeddings first. Unlike the main pipeline, these baselines
    don't rely on the pre-trained word2vec embeddings.
 * `BaselinesTfIdf` computes the tf--idf baseline.
 * `Pretty Plots` can be used to load in JSON data saved from TensorBoard
   and compute the plot used in the report. For maximum reproducibility,
   the original JSON dumps have been checked into the repository, since
   they're quite small anyway.


## License

Copyright 2016, The project authors.
Code licensed under the Apache License, Version 2.0.


[0]:https://inclass.kaggle.com/c/cil-text-classification/data
[1]:https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/

