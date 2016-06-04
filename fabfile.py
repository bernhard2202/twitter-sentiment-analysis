# Simple hacky script to train our pipeline on Google Compute Engine.
# TODO(andrei): Use 'screen' in case connection dies.
# TODO(andrei): Auto-download results.
# TODO(andrei): Support AWS.

from __future__ import with_statement

import os

from fabric.api import *
from fabric.contrib.console import confirm
from fabric.contrib.project import rsync_project as rsync

# TODO(andrei): Read from file.

env.hosts = [
    # GCE (Andrei)
    '104.154.88.35'
]

env.key_filename = '~/.ssh/google_compute_engine'

def train():
    # If something stops working, make sure you're 'rsync'ing everything you
    # need to the remote host!

    print("Will train TF model remotely.")

    run('mkdir -p ~/deploy/data/preprocessing')

    folder = os.path.join('data', 'preprocessing') + '/'
    data_files = ['vocab.pkl', 'vocab-inv.pkl', 'embeddings.npy', 'trainX.npy',
                    'trainY.npy']
    # This does no tilde expansion, and this is what we want.
    remote_folder = os.path.join('~/deploy', folder)

    # This syncs the data (needs to be preprocessed in advance).
    rsync(local_dir=folder, remote_dir=remote_folder,
          exclude=['*.txt'])

    put(local_path='./train_model.py',
        remote_path=os.path.join('~/deploy', 'train_model.py'))

    # This syncs the model code.
    rsync(local_dir='model', remote_dir='deploy')
    with cd('deploy'):
        run('python -m train_model --num_epochs 1' \
            '--batch_size 256 --evaluate_every 250' \
            '--checkpoint_every 1000 --output_every 50')

    local('mkdir -p data/runs/gce')

    # This downloads the pipeline output.
    get(remote_path='~/deploy/data/runs', local_path='data/runs/gce')

    print("Uploaded data.")

def host_type():
    """An example of a Fabric command."""

    # This runs on your machine.
    local('uname -a')

    # This runs on the remote host(s) specified by the -H flag. If none
    # specified, this runs on all 'env.hosts'.
    run('uname -a && lsb_release -a')

