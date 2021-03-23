import os
import logging, argparse
import itertools
import torch

from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel
from models import LexNet

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import time

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")

    parser.add_argument("--odir", default = "training_output")
    parser.add_argument("--tag", default = "test")

    parser.add_argument("--criterion", default = "ce")
    parser.add_argument("--input_shape", default = "1280,64")
    parser.add_argument("--weights", default = "None")
    parser.add_argument("--device", default = "cuda")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.mkdir(args.odir)
            logging.debug('root: made output directory {0}'.format(args.odir))

    return args

def main():
    args = parse_args()

    if args.criterion == 'ce':
        criterion = CrossEntropyLoss()
        o_activation = 'softmax'

    input_shape = tuple(map(int, args.input_shape.split(',')))
    model = LexNet(input_shape)
    device = torch.device(args.device)

    model = model.to(device)

    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location=device)
        checkpoint_ = dict()

        for key in checkpoint.keys():
            if 'module.' in key:
                checkpoint_[key.replace('module.', '')] = checkpoint[key]
            else:
                checkpoint_[key] = checkpoint[key]

        model.load_state_dict(checkpoint_)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, factor=float(args.rl_factor), patience=int(args.n_plateau))

    min_val_loss = np.inf
    val_accuracy = np.inf

    val_count = 0

    history = dict()
    history['epoch'] = []
    history['loss'] = []
    history['time'] = []
    history['accuracy'] = []
    history['val_loss'] = []
    history['val_accuracy'] = []

    for epoch in range(int(args.n_epochs)):
        t1 = time.time()

        losses = []
        accuracies = []

        model.train()

        for i, (x, y) in enumerate(training_generator):
            x = torch.FloatTensor(x)
            y = torch.LongTensor(y)

            x = x.to(device)
            y = torch.squeeze(y.to(device))

            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            y_pred = np.argmax(y_pred, axis=1)
            accuracies.append(accuracy_score(y, y_pred))

            losses.append(loss.item())

            if (i + 1) % 100 == 0:
                logging.info(
                    'root: Epoch {0}, step {3}: got loss of {1}, acc: {2}'.format(epoch, np.mean(losses),
                                                                                  np.mean(accuracies), i + 1))

        history['epoch'].append(epoch)
        history['loss'].append(np.mean(losses))
        history['accuracy'].append(np.mean(accuracies))

        logging.info('this epochs training took {} seconds'.format(time.time() - t1))
        history['time'].append(time.time() - t1)

        optimizer.zero_grad()

        # need to do this sometimes
        model.eval()

        logging.info('root: Epoch {0}, got loss of {1}, acc: {2}'.format(epoch, np.mean(losses), np.mean(accuracies)))

        val_losses = []
        val_accuracies = []
        for i, (x, y) in enumerate(validation_generator):
            x = torch.FloatTensor(x)
            y = torch.LongTensor(y)

            x = x.to(device)
            y = torch.squeeze(y.to(device))

            y_pred = model(x)

            loss = criterion(y_pred, y)

            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            y_pred = np.argmax(y_pred, axis=1)
            val_accuracies.append(accuracy_score(y, y_pred))

            val_losses.append(loss.item())

        val_loss = np.mean(val_losses)

        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(np.mean(val_accuracies))

        logging.info(
            'root: Epoch {0}, got val loss of {1}, acc: {2} '.format(epoch, val_loss, np.mean(val_accuracies)))

        if val_loss < min_val_loss:
            logging.debug('saving weights...')
            torch.save(model.state_dict(), os.path.join(args.odir, '{0}.weights'.format(args.tag)))
            logging.debug('saved weights')

            min_val_loss = copy.copy(val_loss)
            val_accuracy = np.mean(val_accuracies)
            val_count = 0
        else:
            val_count += 1

        if val_count >= int(args.n_early):
            break

        scheduler.step(val_loss)

    np.savetxt(os.path.join(args.odir, '{0}.evals'.format(args.tag)),
               np.array([min_val_loss, val_accuracy], dtype=np.float32))

    # save history of training
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(args.odir, '{0}_history.csv'.format(args.tag)), index=False)
