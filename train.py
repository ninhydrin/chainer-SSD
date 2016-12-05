#!/usr/bin/env python
# from __future__ import print_function
import argparse
import os
import sys
import threading
import time

import numpy as np
import queue

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import ssd_net


parser = argparse.ArgumentParser(
    description='Learning SSD')
parser.add_argument('train', help='Path to training image-label list file')
parser.add_argument('val', help='Path to validation image-label list file')
parser.add_argument('--mean', '-m', default='',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--batchsize', '-B', type=int, default=100,
                    help='Learning minibatch size')
parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                    help='Validation minibatch size')
parser.add_argument('--epoch', '-E', default=30, type=int,
                    help='Number of epochs to learn')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--loaderjob', '-j', default=20, type=int,
                    help='Number of parallel data loading processes')
parser.add_argument('--train_root', '-tr', default='.',
                    help='Root directory path of train image files')
parser.add_argument('--val_root', '-vr', default='.',
                    help='Root directory path of val image files')
parser.add_argument('--out', '-o', default='random_break_conv_fix.model',
                    help='Path to save model on each validation')
parser.add_argument('--outstate', '-s', default='state',
                    help='Path to save optimizer state on each validation')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', default='',
                    help='Resume the optimization from snapshot')

parser.set_defaults(test=False)
args = parser.parse_args()


def model_resume():
    if args.initmodel:
        print('Load model from', args.initmodel)
        serializers.load_npz(args.initmodel, model)
    if args.resume:
        print('Load optimizer state from', args.resume)
        serializers.load_npz(args.resume, optimizer)


def model_save(name):
    serializers.save_npz(args.out.format(name), model)
    serializers.save_npz(args.outstate.format(name), optimizer)


def load_image_list(path, root):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append((os.path.join(root, pair[0]), np.int32(pair[1])))
    return tuples

assert 50000 % args.val_batchsize == 0

denominator = 100000

train_list = load_image_list(args.train, args.tr)
val_list = load_image_list(args.val, args.vr)
mean_image = np.load(args.mean) if args.mean else np.array([104, 117, 123])

model = ssd_net.SSD()
optimizer = optimizers.MomentumSGD(lr=0.0001, momentum=0.9)
optimizer.setup(model)
# optimizer.add_hook(chainer.optimizer.WeightDecay(0.01))

model_resume()
data_q = queue.Queue(maxsize=1)
res_q = queue.Queue()

cropwidth = 256 - model.insize

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np


class Trainer:
    def __init__(self, model, data_q, res_q, args):
        self.model = model
        self.data_q = data_q
        self.res_q = res_q
        self.args = args

    def train_loop(self):
        # Trainer
        val_count = 0
        while True:
            while data_q.empty():
                time.sleep(0.1)
            inp = self.data_q.get()
            if inp == 'end':  # quit
                self.res_q.put('end')
                break
            elif inp == 'train':  # restart training
                self.res_q.put('train')
                model.train = True
                continue
            elif inp == 'val':  # start validation
                self.res_q.put('val')
                model_save(val_count * denominator)
                model.train = False
                continue

            volatile = 'off' if model.train else 'on'
            x = chainer.Variable(xp.asarray(inp[0]), volatile=volatile)
            t = chainer.Variable(xp.asarray(inp[1]), volatile=volatile)

            if self.model.train:
                optimizer.update(self.model, x, t)
            else:
                self.model(x, t)

            self.res_q.put(
                (float(self.model.loss.data), float(self.model.accuracy.data)))
            del x, t

# Invoke threads

feeder = threading.Thread(target=feed_data)
feeder.daemon = True
feeder.start()
logger = threading.Thread(target=log_result)
logger.daemon = True
logger.start()

train_loop()
feeder.join()
logger.join()

# Save final model
model_save("final")
