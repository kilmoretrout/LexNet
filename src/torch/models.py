import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout, Conv1d, AvgPool1d

import os
import numpy as np
import logging, argparse
import pandas
import torch.optim as optim
import random

from torch.utils.data import Dataset
import h5py

import random

# function for returning an MLP
# given a list of channel sizes
def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]), Dropout(0.25))
        for i in range(1, len(channels))
    ])

class H5PyCategoricalDataset(Dataset):
    def __init__(self, ifile, buffer_size = 25, chunk_size = 4, shuffle = True):
        super(H5PyCategoricalDataset, self).__init__()
        # buffer size (how many chunks to load) at a time
        # chunk size (the number of replicates in each "chunk" i.e. each H5Py dataset in the file)
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle

        # the h5py object where we'll find all the data
        self.ifile = h5py.File(ifile, 'r')
        self.class_names = sorted(list(ifile.keys()))

        self.load_keys()
        self.length = len(self.keys()) * chunk_size

        self.load_buffer()

    def load_keys(self):
        self.keys = []
        for class_name in self.class_names:
            self.keys.extend([class_name + '/{}'.format(u) for u in self.ifile[class_name].keys()])

        if self.shuffle:
            random.shuffle(self.keys())

    def load_buffer(self):
        self.buffer = []
        self.buffer_y = []


        if len(self.keys) < self.buffer_size:
            self.load_keys()
        ix = np.random.choice(range(len(self.keys)), self.buffer_size, replace = False)
        ix = sorted(ix, reverse = True)
        keys = [self.keys[u] for u in ix]

        for ii in ix:
            del self.keys[ii]

        for key in keys:
            self.buffer.append(self.ifile[key]['x_0'])

            index = self.class_names.index(key.split('/')[0])
            self.buffer_y.append(np.repeat(index, (self.chunk_size, )))

        self.buffer = np.vstack(self.buffer)
        self.buffer_y = np.hstack(self.buffer_y)

    def __len__(self):
        return self.length

    def __getitem__(self, ix):
        if len(self.buffer) == 0:
            self.load_buffer()

        ret = (torch.FloatTensor(self.buffer[-1]), torch.LongTensor(self.buffer_y[-1]))

        del self.buffer[-1]
        del self.buffer_y[-1]

        return ret

class LexNet(torch.nn.Module):
    def __init__(self, input_shape = (1280, 64), n_classes = 3):
        super(LexNet, self).__init__()
        self.c1 = Conv1d(64, 256, kernel_size = 2)
        l = input_shape[0]
        l -= 1

        self.c2 = Conv1d(256, 128, kernel_size = 2)
        l -= 1

        self.pool = AvgPool1d(kernel_size = 2)
        l = int(np.floor(((l - 2) / 2.) + 1))

        self.d1 = Dropout(0.25)

        self.c3 = Conv1d(128, 128, kernel_size = 2)
        l -= 1
        self.d2 = Dropout(0.25)

        l = int(np.floor(((l - 2) / 2.) + 1))
        self.mlp = MLP([l * 128, 128, 128])
        self.final = Lin(128, 3)
        self.soft = nn.Softmax()

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = self.pool(x)

        x = self.d1(x)

        x = torch.relu(self.c3(x))
        x = self.pool(x)

        x = self.d2(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])

        x = self.mlp(x)
        x = self.soft(self.final(x))

        return x

# testing, testing, 1, 2, 3,....
if __name__ == '__main__':
    model = LexNet()

    x = torch.randn((100, 64, 1280))
    x = model.forward(x)

    print(x.shape)