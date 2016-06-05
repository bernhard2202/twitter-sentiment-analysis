"""Fabric deployment file for remote model training.

TODO(andrei): fabric is deprecated. Use 'pyinvoke'.

Uses Python Fabric (http://www.fabfile.org/).

Make sure that 'env.hosts' points to wherever you want to train your model, and
that the remote host has tensorflow installed.

Examples:
    `fab train`        starts to train the TF model on the remote host.
    `fab tb`           starts Tensorboard on the remote host.
    `fab latest_tb`    starts Tensorboard on the remote host, configuring it to
                       only display the results of the most recent run.
"""

# TODO(andrei): Support AWS and Euler. Euler will probably not allow live
# tensorboard preview, but that's OK.

from __future__ import with_statement

import os

from fabric.api import *
from fabric.contrib.project import rsync_project as rsync

# TODO(andrei): Read from file.

env.hosts = [
    # GCE (Andrei)
    '104.154.88.35'
]

env.key_filename = '~/.ssh/google_compute_engine'


def latest_run_id():
    # TODO(andrei): Nicer way of doing this?
    return "ls -t ~/deploy/data/runs | cat | head -n1"


def train():
    # If something stops working, make sure you're 'rsync'ing everything you
    # need to the remote host!

    # TODO(andrei): Use 'screen' in case connection dies.
    # TODO(andrei): Auto-download results.

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

    print("Uploaded data and code.")
    print("Starting to train.")

    with cd('deploy'):
        # TODO(andrei): Pass these parameters as arguments to fabric.
        run('python -m train_model --num_epochs 20'
            ' --batch_size 256 --evaluate_every 250'
            ' --checkpoint_every 1000 --output_every 100')

    local('mkdir -p data/runs/gce')

    # This downloads the pipeline output.
    # TODO(andrei): Nicer folder structure
    run('mkdir -p /tmp/last_tf_run')
    run('cp -R ~/deploy/data/runs/$({})/ /tmp/last_tf_run'.format(latest_run_id()),
        shell_escape=False, shell=False)
    get(remote_path='/tmp/last_tf_run/*',
        local_path='data/runs/gce')
    print("Downloaded the pipeline results.")


def tb():
    """See: 'tensorboard'"""
    return tensorboard()


def tensorboard():
    """Starts a remote tensorboard to see your pipeline's status.

    TODO(andrei): Run in screen.

    Make sure you allow TCP on port 6006 for the remote machine!
    """

    with cd('deploy'):
        tb_cmd = 'tensorboard --logdir data/runs'
        screen = "screen -dmS tensorboard_screen bash -c '{}'".format(tb_cmd)
        print(screen)
        run(screen, pty=False)


def latest_tb():
    """See: 'latest_tensorboard'"""
    return latest_tensorboard()


def latest_tensorboard():
    """
    Uses the latest log dir as a source.
    """
    with cd('deploy'):
        # This sets logdir to the most recent run.
        run('tensorboard --logdir data/runs/$(ls -t data/runs | cat | head -n1)/summaries',
            shell_escape=False, shell=False)


def kill_tb():
    return kill_tensorboard()


def kill_tensorboard():
    run('killall tensorboard')


def host_type():
    """An example of a Fabric command."""

    # This runs on your machine.
    local('uname -a')

    # This runs on the remote host(s) specified by the -H flag. If none are
    # specified, this runs on all 'env.hosts'.
    run('uname -a && lsb_release -a')
    run('pwd')
    with cd('/tmp'):
        run('pwd')

