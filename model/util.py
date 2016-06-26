"""Helpers and utilities."""

import numpy as np


def batch_iter(data, batch_size, num_epochs):
    """ Generates a batch iterator for a data set."""
    data_size = len(data)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs): # Shuffle the data at each epoch
        shuffled_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffled_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

