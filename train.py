#!/usr/bin/env python
# from __future__ import print_function
import argparse
import os
import sys
import threading
import time
import pickle

import numpy as np
import queue

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import ssd_net


parser = argparse.ArgumentParser(
    description='Learning SSD')
parser.add_argument('--train', help='year of training image set', default="2007", choices=("2007", "2012", "2712"))
parser.add_argument('--val', help='year of validation image set', default="2007", choices=("2007", "2012"))
parser.add_argument('--mean', '-m', default='',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--batchsize', '-B', type=int, default=14,
                    help='Learning minibatch size')
parser.add_argument('--val_batchsize', '-b', type=int, default=14,
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
parser.add_argument('--out', '-o', default='model/ssd_{}.model',
                    help='Path to save model on each validation')
parser.add_argument('--outstate', '-s', default='state/ssd_{}.state',
                    help='Path to save optimizer state on each validation')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--log', default='log/ssd.log',
                    help='log file name')


parser.set_defaults(test=False)
args = parser.parse_args()

if not os.path.isdir("model"):
    os.mkdir("model")
if not os.path.isdir("state"):
    os.mkdir("state")


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
#assert 50000 % args.val_batchsize == 0

denominator = 10

train_path = "trainval_voc{}.pkl".format(args.train)
val_path = "test_voc{}.pkl".format(args.val)
train_list = pickle.load(open(train_path, "rb")) 
val_list = pickle.load(open(val_path, "rb"))

mean_image = np.load(args.mean) if args.mean else np.array([104, 117, 123])

model = ssd_net.SSD()
optimizer = optimizers.MomentumSGD(lr=0.00001, momentum=0.9)
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


import feeder
import logger
import batch_sampler
feed_data = feeder.Feeder(model.mbox_prior, train_list, val_list, mean_image, batch_sampler.batch_sampler, data_q, args)
log_result = logger.Logger(args.log, res_q, args)

path, size, bb = train_list[0]
l, bb = bb[:,0],bb[:,1:]
s = feed_data.sampler(size, bb)
size, bb = s[0]
a = feed_data.reader(path, size, bb, l)
k = chainer.Variable(a[0][np.newaxis, :])
k.to_gpu()
model.train = True
model(k, [a[4]],[a[3]],[a[2]],[a[1]])
sys.exit()

class Trainer:
    def __init__(self, model, data_q, res_q, args):
        self.model = model
        self.data_q = data_q
        self.res_q = res_q
        self.args = args

    def __call__(self):
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
            img, loc_mask, conf_mask, loc, conf = inp
            volatile = 'off' if model.train else 'on'
            x = chainer.Variable(xp.asarray(img), volatile=volatile)
            t_conf = chainer.Variable(xp.asarray(conf), volatile=volatile)
            t_loc = chainer.Variable(xp.asarray(loc), volatile=volatile)

            if self.model.train:
                optimizer.update(self.model, x, t_conf, t_loc, conf_mask, loc_mask)
            else:
                self.model(x, t_conf, t_loc, conf_mask, loc_mask)

            self.res_q.put(
                (float(self.model.loss.data), float(self.model.accuracy.data)))
            del x, t_conf, t_loc


feeder = threading.Thread(target=feed_data)
feeder.daemon = True
feeder.start()

logger = threading.Thread(target=log_result)
logger.daemon = True
logger.start()

trainer = Trainer(model, data_q, res_q, args)
trainer()
feeder.join()
logger.join()

# Save final model
model_save("final")

