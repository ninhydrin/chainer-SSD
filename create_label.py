import pickle

import numpy as np

import bbox
import ssd_net

model = ssd_net.SSD()
data = pickle.load(open("test_voc2007.pkl", "rb"))


class TrainMaker:
    def __init__(self, prior, datas, insize=300):
        self.prior = prior
        self.datas = datas
        self.insize = insize
        self.prior_num = self.prior.shape[0]

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
