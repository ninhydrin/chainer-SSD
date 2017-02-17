import pickle
import multiprocessing

import numpy as np
from PIL import Image
from scipy.misc import imresize

import util.bbox as bbox
import util.ssd as ssd
import batch_sampler
from sampler import Sampler


class Reader:
    def __init__(self, prior, mean, insize=300):
        self.prior = prior
        self.insize = insize
        self.prior_num = self.prior.shape[0]
        self.mean = mean

    def __call__(self, path, size, BB, label, flip=False):
        image = np.asarray(Image.open(path).convert("RGB"))
        left, top, right, bottom = size
        image = image[top:bottom, left:right, :].astype(np.float32)
        h, w = image.shape[:2]
        image -= self.mean
        BB[:, (0, 2)] -= left
        BB[:, (1, 3)] -= top
        if (BB < 0).any():
            assert not (BB < 0).any()
        BB[:, (0, 2)] /= w
        BB[:, (1, 3)] /= h
        if flip and np.random.randint(2) == 3:
            image = image[:, :, ::-1]
            BB[:, 0], BB[:, 2] = w - BB[:, 2], right - BB[:, 0]
        image = imresize(image, (self.insize, self.insize), interp='bicubic')
        loc_mask, conf_mask, loc, conf = self.make_mask(BB, label)
        image = image.transpose(2, 0, 1).astype(np.float32)
        return image, loc_mask, conf_mask, loc, conf

    def make_mask(self, BB, label):
        loc_mask = np.zeros([self.prior_num, 4], dtype=np.float32)
        conf_mask = np.zeros([self.prior_num, 21], dtype=np.float32)
        overlap = bbox.bbox_overlaps2(self.prior[:, 0], BB)
        positions = np.array(np.where(overlap > 0.5))[0]
        loc_mask[positions] = 1
        conf_mask[positions] = 1

        which_class = overlap.argmax(1)
        loc = np.zeros([self.prior_num, 4])
        loc[positions] = BB[which_class[positions]]
        conf = np.tile(20, self.prior_num).astype(np.int32)
        conf[positions] = label[which_class[positions]]
        loc = ssd.encoder(loc, self.prior)
        loc[np.where(loc_mask == 0)[0]] = 0
        if not np.all(conf.max() < 21):
            print(BB)
            print(overlap.shape)
            print("conf max = ", conf.max())
            assert conf.max() < 21
        return [loc_mask, conf_mask, loc, conf]


class Feeder:
    def __init__(self, prior, train_list, val_list, mean, batch_sampler, data_q, args, denominator=100000, insize=300):
        self.prior = prior
        self.train_list = train_list
        self.val_list = val_list
        self.insize = insize
        self.prior_num = self.prior.shape[0]
        self.mean = mean
        self.args = args
        self.data_q = data_q
        self.denominator = denominator
        self.sampler = Sampler(batch_sampler)
        self.reader = Reader(prior, mean)

    def __call__(self):
    # Data feeder
        args = self.args
        i = 0
        count = 0
        img_batch = np.ndarray(
            (args.batchsize, 3, self.insize, self.insize), dtype=np.float32)
        conf_batch = np.ndarray(
            (args.batchsize, self.prior_num), dtype=np.int32)
        loc_batch = np.ndarray(
            (args.batchsize, self.prior_num, 4), dtype=np.float32)
        conf_mask = np.ndarray(
            (args.batchsize, self.prior_num, 21), dtype=np.float32)
        loc_mask = np.ndarray(
            (args.batchsize, self.prior_num, 4), dtype=np.float32)

        val_img_batch = np.ndarray(
            (args.val_batchsize, 3, self.insize, self.insize), dtype=np.float32)
        val_conf_batch = np.ndarray(
            (args.val_batchsize, self.prior_num), dtype=np.int32)
        val_loc_batch = np.ndarray(
            (args.val_batchsize, self.prior_num, 4), dtype=np.float32)
        val_conf_mask = np.ndarray(
            (args.val_batchsize, self.prior_num, 21), dtype=np.float32)
        val_loc_mask = np.ndarray(
            (args.val_batchsize, self.prior_num, 4), dtype=np.float32)

        batch_pool = [None] * args.batchsize
        val_batch_pool = [None] * args.val_batchsize
        pool = multiprocessing.Pool(args.loaderjob)
        self.data_q.put('train')

        for epoch in range(1, 1 + args.epoch):
            perm = np.random.permutation(len(self.train_list))
            for idx in perm:
                path, size, BB = self.train_list[idx]
                label = BB[:, 0]
                BB = BB[:, 1:]
                sample = self.sampler(size, BB)
                for sample_size, sample_bbox in sample:
                    batch_pool[i] = pool.apply_async(self.reader, (path, sample_size, sample_bbox, label, True))
                    i += 1
                    if i == args.batchsize:
                        for j, x in enumerate(batch_pool):
                            img_batch[j], loc_mask[j], conf_mask[j], loc_batch[j], conf_batch[j] = x.get()
                        self.data_q.put((img_batch.copy(), loc_mask.copy(),
                                         conf_mask.copy(), loc_batch.copy(), conf_batch.copy()))
                        i = 0
                    count += 1
                    if count % self.denominator == 0:
                        self.data_q.put('val')
                        j = 0
                        for path, size, BB in self.val_list:
                            size = [0, 0, size[1], size[0]]
                            label = BB[:, 0]
                            BB = BB[:, 1:]
                            val_batch_pool[j] = pool.apply_async(self.reader, (path, size, BB, label, False))
                            j += 1
                            if j == args.val_batchsize:
                                for k, x in enumerate(val_batch_pool):
                                    val_img_batch[k], val_loc_mask[k], val_conf_mask[k], val_loc_batch[k], val_conf_batch[k] = x.get()
                                self.data_q.put((val_img_batch.copy(), val_loc_mask.copy(),
                                                 val_conf_mask.copy(), val_loc_batch.copy(), val_conf_batch.copy()))
                                j = 0
                        self.data_q.put('train')
        pool.close()
        pool.join()
        self.data_q.put('end')
