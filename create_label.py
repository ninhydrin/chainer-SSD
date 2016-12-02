import pickle

import numpy as np

import bbox
import ssd_net

model = ssd_net.SSD()
data = pickle.load(open("test_voc2007.pkl", "rb"))
cropwidth = 256 - model.insize


class Feeder:

    def __init__(self, prior, datas, insize=300):
        self.prior = prior
        self.datas = datas
        self.insize = insize
        self.prior_num = self.prior.shape[0]

    def read_image(self, data, center=False, flip=False):
        path, size, BBs = data
        image = np.asarray(Image.open(path).convert("RGB").resize((self.insize, self.insize)))
        image = image.transpose(2, 0, 1)
        if center:
            top = left = cropwidth / 2
        else:
            top = np.random.randint(0, cropwidth - 1)
            left = np.random.randint(0, cropwidth - 1)
        bottom = model.insize + top
        right = model.insize + left
        image = image[:, top:bottom, left:right].astype(np.float32)
        image -= mean_image[:, top:bottom, left:right]
        # image /= 255
        if flip and random.randint(0, 1) == 0:
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
        i = 0
        count = 0
        x_batch = np.ndarray(
            (args.batchsize, 3, model.insize, model.insize), dtype=np.float32)
        y_batch = np.ndarray((args.batchsize,), dtype=np.int32)
        val_x_batch = np.ndarray(
            (args.val_batchsize, 3, model.insize, model.insize), dtype=np.float32)
        val_y_batch = np.ndarray((args.val_batchsize,), dtype=np.int32)

        batch_pool = [None] * args.batchsize
        val_batch_pool = [None] * args.val_batchsize
        pool = multiprocessing.Pool(args.loaderjob)
        data_q.put('train')

        for epoch in range(1, 1 + args.epoch):
            perm = np.random.permutation(len(train_list))
            for idx in perm:
                data = train_list[idx]
                batch_pool[i] = pool.apply_async(self.read_image, (data, False, True))
                i += 1

                if i == args.batchsize:
                    for j, x in enumerate(batch_pool):
                        x_batch[j], y_batch[j] = x.get()
                    data_q.put((x_batch.copy(), y_batch.copy()))
                    i = 0

                count += 1
                if count % denominator == 0:
                    data_q.put('val')
                    j = 0
                    for data in val_list:
                        val_batch_pool[j] = pool.apply_async(
                            read_image, (data, True, False))
                        j += 1
                        if j == args.val_batchsize:
                            for k, x in enumerate(val_batch_pool):
                                val_x_batch[k], val_y_batch[k] = x.get()
                            data_q.put((val_x_batch.copy(), val_y_batch.copy()))
                            j = 0

                    data_q.put('train')
        pool.close()
        pool.join()
        data_q.put('end')
