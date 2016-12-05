import pickle
import multiprocessing

import numpy as np
from PIL import Image

import bbox

data = pickle.load(open("test_voc2007.pkl", "rb"))
cropwidth = 256 - model.insize


class Feeder:

    def __init__(self, prior, train_list, val_list, mean, args, data_q, denominator=100000, insize=300):
        self.prior = prior
        self.train_list = train_list
        self.val_list = val_list
        self.insize = insize
        self.prior_num = self.prior.shape[0]
        self.mean = mean
        self.args = args
        self.data_q = data_q
        self.denominator = denominator

    def change_aspect(self, size, BB):
        aspect_h = 0.5 + np.random.random() * 1.5
        aspect_w = 0.5 + np.random.random() * 1.5
        size[0] *= aspect_h
        size[1] *= aspect_w
        BB[:, (0, 2)] *= aspect_w
        BB[:, (1, 3)] *= aspect_h

    def read_image(self, data, center=False, flip=False):
        path, size, BBs = data
        image = np.asarray(Image.open(path).convert("RGB").resize((self.insize, self.insize)))
        image = image.transpose(2, 0, 1)
        if center:
            top = left = cropwidth / 2
        else:
            top = np.random.randint(0, cropwidth - 1)
            left = np.random.randint(0, cropwidth - 1)
        bottom = self.insize + top
        right = self.insize + left
        image = image[:, top:bottom, left:right].astype(np.float32)
        image -= self.mean[:, top:bottom, left:right]
        # image /= 255
        if flip and np.random.randint(2) == 0:
            return image[:, :, ::-1]
        else:
            return image

    def make_sample(self, data):
        path, size, BBs = data
        h, w = size
        label = BBs[:, 0]

        BB = BBs[:, 1:].copy()
        BB[:, (0, 2)] *= self.insize / w
        BB[:, (1, 3)] *= self.insize/h
        loc_mask = np.zeros([self.prior_num, 4])
        conf_mask = np.zeros([self.prior_num, 21])
        overlap = bbox.bbox_overlaps2(self.prior[:, 0] * self.insize, BB)
        positions = np.array(np.where(overlap > 0.5))
        conf_mask[positions[0]] = 1
        loc_mask[positions[0]] = 1

        conf_t = overlap.argmax(1)
        loc_t = np.zeros([self.prior_num, 4])
        loc_t = np.array([self.encoder(BB[conf_t[i]] / self.insize, self.prior[i])
                          if loc_mask[i][0] else [0, 0, 0, 0]
                          for i in range(self.prior_num)])
        conf_t = [label[conf_t[i]] for i in range(self.prior_num)]
        conf_t *= conf_mask[:, 0]
        conf_t += conf_mask[:, 0]
        return ([loc_mask, conf_mask, loc_t, conf_t])

    def encoder(self, bbox, prior):
        prior_bbox = prior[0]
        prior_variance = prior[1]
        encode_bbox = np.array([0] * 4, dtype=np.float32)
        prior_width = prior_bbox[2] - prior_bbox[0]
        prior_height = prior_bbox[3] - prior_bbox[1]
        prior_center_x = (prior_bbox[0] + prior_bbox[2]) / 2.
        prior_center_y = (prior_bbox[1] + prior_bbox[3]) / 2.
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        if bbox_width <= 0 or bbox_height <= 0:
            return encode_bbox
        bbox_center_x = (bbox[0] + bbox[2]) / 2.
        bbox_center_y = (bbox[1] + bbox[3]) / 2.
        encode_bbox[0] = (bbox_center_x - prior_center_x) / prior_width / prior_variance[0]
        encode_bbox[1] = (bbox_center_y - prior_center_y) / prior_height / prior_variance[1]
        encode_bbox[2] = np.log(bbox_width / prior_width) / prior_variance[2]
        encode_bbox[3] = np.log(bbox_height / prior_height) / prior_variance[3]
        return encode_bbox

    def feed_data(self):
    # Data feeder
        args = self.args
        i = 0
        count = 0
        x_batch = np.ndarray(
            (args.batchsize, 3, self.insize, self.insize), dtype=np.float32)
        y_batch = np.ndarray((args.batchsize,), dtype=np.int32)
        val_x_batch = np.ndarray(
            (args.val_batchsize, 3, self.insize, self.insize), dtype=np.float32)
        val_y_batch = np.ndarray((args.val_batchsize,), dtype=np.int32)

        batch_pool = [None] * args.batchsize
        val_batch_pool = [None] * args.val_batchsize
        pool = multiprocessing.Pool(args.loaderjob)
        self.data_q.put('train')

        for epoch in range(1, 1 + args.epoch):
            perm = np.random.permutation(len(self.train_list))
            for idx in perm:
                data = self.train_list[idx]
                batch_pool[i] = pool.apply_async(self.read_image, (data, False, True))
                i += 1

                if i == args.batchsize:
                    for j, x in enumerate(batch_pool):
                        x_batch[j], y_batch[j] = x.get()
                    self.data_q.put((x_batch.copy(), y_batch.copy()))
                    i = 0

                count += 1
                if count % self.denominator == 0:
                    self.data_q.put('val')
                    j = 0
                    for data in self.val_list:
                        val_batch_pool[j] = pool.apply_async(
                            self.read_image, (data, True, False))
                        j += 1
                        if j == args.val_batchsize:
                            for k, x in enumerate(val_batch_pool):
                                val_x_batch[k], val_y_batch[k] = x.get()
                            self.data_q.put((val_x_batch.copy(), val_y_batch.copy()))
                            j = 0

                    self.data_q.put('train')
        pool.close()
        pool.join()
        self.data_q.put('end')

class Sampler:
    class BatchSampler:
        def __init__(self, max_scale, min_scale, max_ar, min_ar, min_jacc, max_trial, max_sample):
            self.max_scale = max_scale
            self.min_scale = min_scale
            self.max_ar = max_ar
            self.min_ar = min_ar
            self.min_jacc = min_jacc
            self.max_trial = max_trial
            self.max_sample = max_sample

    def __init__(self, batch_sampler,):
        self.batch_sampler = batch_sampler

    def sample_bbox(self, sampler):
        # TODO 何もしない時

        scale = sampler.min_scale + (sampler.max_scale - sampler.min_scale) * np.random.random()
        

