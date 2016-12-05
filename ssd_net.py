# encoding:utf-8
import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

class SSD (chainer.Chain):
    insize = 300

    def set_info(self, name, h, w, d_num):
        setattr(self, name + "_h", h)
        setattr(self, name + "_w", w)
        setattr(self, name + "_d", d_num)

    def __init__(self):
        super(SSD, self).__init__(
            conv1_1=L.Convolution2D(3,  64, 3, pad=1),
            conv1_2=L.Convolution2D(64,  64, 3, pad=1),
            conv2_1=L.Convolution2D(64, 128,  3, pad=1),
            conv2_2=L.Convolution2D(128, 128,  3, pad=1),
            conv3_1=L.Convolution2D(128, 256,  3, pad=1),
            conv3_2=L.Convolution2D(256, 256,  3, pad=1),
            conv3_3=L.Convolution2D(256, 256,  3, pad=1),

            conv4_1=L.Convolution2D(256, 512,  3, pad=1),
            conv4_2=L.Convolution2D(512, 512,  3, pad=1),
            conv4_3=L.Convolution2D(512, 512,  3, pad=1),

            conv5_1=L.Convolution2D(512, 512,  3, pad=1),
            conv5_2=L.Convolution2D(512, 512,  3, pad=1),
            conv5_3=L.Convolution2D(512, 512,  3, pad=1),

            fc6=L.DilatedConvolution2D(512, 1024,  3, pad=6, dilate=6),
            fc7=L.Convolution2D(1024, 1024, 1),

            conv6_1=L.Convolution2D(1024, 256,  1),
            conv6_2=L.Convolution2D(256, 512,  3, stride=2, pad=1),

            conv7_1=L.Convolution2D(512, 128,  1),
            conv7_2=L.Convolution2D(128, 256,  3, stride=2, pad=1),

            conv8_1=L.Convolution2D(256, 128,  1),
            conv8_2=L.Convolution2D(128, 256,  3, stride=2, pad=1),

            normalize=L.Scale(W_shape=512),


            conv4_3_norm_mbox_loc = L.Convolution2D(512, 12,  3, pad=1), #3 prior boxes
            conv4_3_norm_mbox_conf = L.Convolution2D(512, 63,  3, pad=1),

            fc7_mbox_loc = L.Convolution2D(1024, 24, 3, pad=1), #6 prior boxes
            fc7_mbox_conf = L.Convolution2D(1024, 126, 3, pad=1),

            conv6_2_mbox_loc = L.Convolution2D(512, 24, 3, pad=1), #6 prior boxes
            conv6_2_mbox_conf = L.Convolution2D(512, 126, 3, pad=1),

            conv7_2_mbox_loc = L.Convolution2D(256, 24, 3, pad=1), #6 prior boxes
            conv7_2_mbox_conf = L.Convolution2D(256, 126, 3, pad=1),

            conv8_2_mbox_loc = L.Convolution2D(256, 24, 3, pad=1), #6 prior boxes
            conv8_2_mbox_conf = L.Convolution2D(256, 126, 3, pad=1),

            pool6_mbox_loc = L.Convolution2D(256, 24, 3, pad=1),
            pool6_mbox_conf = L.Convolution2D(256, 126, 3, pad=1), #6 prior boxes

        )
        self.train = False
        self.set_info("c4", 38, 38, 3)
        self.set_info("f7", 19, 19, 6)
        self.set_info("c6", 10, 10, 6)
        self.set_info("c7", 5, 5, 6)
        self.set_info("c8", 3, 3, 6)
        self.set_info("p6", 1, 1, 6)

        self.conv4_3_norm_mbox_priorbox = self.prior((38, 38), 30., 0, [2], 1, 1, (0.1, 0.1, 0.2, 0.2))
        self.fc7_mbox_priorbox = self.prior((19, 19), 60., 114., [2, 3], 1, 1, (0.1, 0.1, 0.2, 0.2))
        self.conv6_2_mbox_priorbox = self.prior((10, 10), 114., 168., [2, 3], 1, 1,(0.1, 0.1, 0.2, 0.2))
        self.conv7_2_mbox_priorbox = self.prior((5, 5),168., 222., [2, 3], 1, 1,(0.1, 0.1, 0.2, 0.2))
        self.conv8_2_mbox_priorbox = self.prior((3, 3), 222., 276., [2, 3], 1, 1,(0.1, 0.1, 0.2, 0.2))
        self.pool6_mbox_priorbox = self.prior((1, 1), 276., 330., [2, 3], 1, 1,(0.1, 0.1, 0.2, 0.2))
        self.mbox_prior = np.concatenate([
            self.conv4_3_norm_mbox_priorbox.reshape(-1, 2, 4),
            self.fc7_mbox_priorbox.reshape(-1, 2, 4),
            self.conv6_2_mbox_priorbox.reshape(-1, 2, 4),
            self.conv7_2_mbox_priorbox.reshape(-1, 2, 4),
            self.conv8_2_mbox_priorbox.reshape(-1, 2, 4),
            self.pool6_mbox_priorbox.reshape(-1, 2, 4),
        ], axis=0)

    def __call__(self, x, t):
        h = F.relu(self.conv1_1(x))
        h = F.max_pooling_2d(F.relu(self.conv1_2(h)), 2, 2)
        h = F.relu(self.conv2_1(h))
        h = F.max_pooling_2d(F.relu(self.conv2_2(h)), 2, 2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.max_pooling_2d(F.relu(self.conv3_3(h)), 2, 2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))

        self.h_conv4_3 = h

        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.max_pooling_2d(F.relu(self.conv5_3(h)), 3, stride=1, pad=1)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        self.h_fc7 = h

        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h))
        self.h_conv6_2 = h

        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h))
        self.h_conv7_2 = h

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        self.h_conv8_2 = h

        h = F.average_pooling_2d(h, 3)
        self.h_pool6 = h

        batchsize, ch, hh, ww = self.h_conv4_3.shape
        kari = F.reshape(self.h_conv4_3, (batchsize*ch, hh*ww))
        kari = F.transpose(kari, (1, 0))
        kari = F.normalize(kari)
        kari = F.transpose(kari, (1, 0))
        kari = F.reshape(kari, (batchsize, ch, hh, ww))

        self.h_conv4_3_norm = self.normalize(kari)
        self.h_conv4_3_norm_mbox_loc = self.conv4_3_norm_mbox_loc(self.h_conv4_3_norm)
        self.h_conv4_3_norm_mbox_conf = self.conv4_3_norm_mbox_conf(self.h_conv4_3_norm)
        self.h_conv4_3_norm_mbox_loc_perm = F.transpose(self.h_conv4_3_norm_mbox_loc,(0,2,3,1))
        self.h_conv4_3_norm_mbox_conf_perm = F.transpose(self.h_conv4_3_norm_mbox_conf,(0,2,3,1))

        self.h_conv4_3_norm_mbox_loc_flat = F.reshape(self.h_conv4_3_norm_mbox_loc_perm, (batchsize, self.c4_h, self.c4_w, self.c4_d, 4))
        self.h_conv4_3_norm_mbox_conf_flat = F.reshape(self.h_conv4_3_norm_mbox_conf_perm, (batchsize, self.c4_h, self.c4_w, self.c4_d, 21))

        self.h_fc7_mbox_loc = self.fc7_mbox_loc(self.h_fc7)
        self.h_fc7_mbox_conf = self.fc7_mbox_conf(self.h_fc7)
        self.h_fc7_mbox_loc_perm = F.transpose(self.h_fc7_mbox_loc,(0,2,3,1))
        self.h_fc7_mbox_conf_perm = F.transpose(self.h_fc7_mbox_conf,(0,2,3,1))
        self.h_fc7_mbox_loc_flat = F.reshape(self.h_fc7_mbox_loc_perm, (batchsize, self.f7_h, self.f7_w, self.f7_d, 4))
        self.h_fc7_mbox_conf_flat = F.reshape(self.h_fc7_mbox_conf_perm, (batchsize, self.f7_h, self.f7_w, self.f7_d, 21))

        self.h_conv6_2_mbox_loc = self.conv6_2_mbox_loc(self.h_conv6_2)
        self.h_conv6_2_mbox_conf = self.conv6_2_mbox_conf(self.h_conv6_2)
        self.h_conv6_2_mbox_loc_perm = F.transpose(self.h_conv6_2_mbox_loc,(0,2,3,1))
        self.h_conv6_2_mbox_conf_perm = F.transpose(self.h_conv6_2_mbox_conf,(0,2,3,1))
        self.h_conv6_2_mbox_loc_flat = F.reshape(self.h_conv6_2_mbox_loc_perm, (batchsize, self.c6_h, self.c6_w, self.c6_d, 4))
        self.h_conv6_2_mbox_conf_flat = F.reshape(self.h_conv6_2_mbox_conf_perm, (batchsize, self.c6_h, self.c6_w, self.c6_d, 21))

        self.h_conv7_2_mbox_loc = self.conv7_2_mbox_loc(self.h_conv7_2)
        self.h_conv7_2_mbox_conf = self.conv7_2_mbox_conf(self.h_conv7_2)
        self.h_conv7_2_mbox_loc_perm = F.transpose(self.h_conv7_2_mbox_loc,(0,2,3,1))
        self.h_conv7_2_mbox_conf_perm = F.transpose(self.h_conv7_2_mbox_conf,(0,2,3,1))
        self.h_conv7_2_mbox_loc_flat = F.reshape(self.h_conv7_2_mbox_loc_perm , (batchsize, self.c7_h, self.c7_w, self.c7_d, 4))
        self.h_conv7_2_mbox_conf_flat = F.reshape(self.h_conv7_2_mbox_conf_perm , (batchsize, self.c7_h, self.c7_w, self.c7_d, 21))

        self.h_conv8_2_mbox_loc = self.conv8_2_mbox_loc(self.h_conv8_2)
        self.h_conv8_2_mbox_conf = self.conv8_2_mbox_conf(self.h_conv8_2)
        self.h_conv8_2_mbox_loc_perm = F.transpose(self.h_conv8_2_mbox_loc,(0,2,3,1))
        self.h_conv8_2_mbox_conf_perm = F.transpose(self.h_conv8_2_mbox_conf,(0,2,3,1))
        self.h_conv8_2_mbox_loc_flat = F.reshape(self.h_conv8_2_mbox_loc_perm, (batchsize, self.c8_h, self.c8_w, self.c8_d, 4))
        self.h_conv8_2_mbox_conf_flat = F.reshape(self.h_conv8_2_mbox_conf_perm, (batchsize, self.c8_h, self.c8_w, self.c8_d, 21))

        self.h_pool6_mbox_loc = self.pool6_mbox_loc(self.h_pool6)
        self.h_pool6_mbox_conf = self.pool6_mbox_conf(self.h_pool6)
        self.h_pool6_mbox_loc_perm = F.transpose(self.h_pool6_mbox_loc,(0,2,3,1))
        self.h_pool6_mbox_conf_perm = F.transpose(self.h_pool6_mbox_conf,(0,2,3,1))
        self.h_pool6_mbox_loc_flat = F.reshape(self.h_pool6_mbox_loc_perm, (batchsize, self.p6_h, self.p6_w, self.p6_d, 4))
        self.h_pool6_mbox_conf_flat = F.reshape(self.h_pool6_mbox_conf_perm, (batchsize, self.p6_h, self.p6_w, self.p6_d, 21))

        self.mbox_loc = F.concat([
            F.reshape(self.h_conv4_3_norm_mbox_loc_flat, [batchsize, -1, 4]),
            F.reshape(self.h_fc7_mbox_loc_flat, [batchsize, -1, 4]),
            F.reshape(self.h_conv6_2_mbox_loc_flat, [batchsize, -1, 4]),
            F.reshape(self.h_conv7_2_mbox_loc_flat, [batchsize, -1, 4]),
            F.reshape(self.h_conv8_2_mbox_loc_flat, [batchsize, -1, 4]),
            F.reshape(self.h_pool6_mbox_loc_flat, [batchsize, -1, 4]),
        ], axis=1)

        self.mbox_conf = F.concat([
            F.reshape(self.h_conv4_3_norm_mbox_conf_flat, [batchsize, -1, 21]),
            F.reshape(self.h_fc7_mbox_conf_flat, [batchsize, -1, 21]),
            F.reshape(self.h_conv6_2_mbox_conf_flat, [batchsize, -1, 21]),
            F.reshape(self.h_conv7_2_mbox_conf_flat, [batchsize, -1, 21]),
            F.reshape(self.h_conv8_2_mbox_conf_flat, [batchsize, -1, 21]),
            F.reshape(self.h_pool6_mbox_conf_flat, [batchsize, -1, 21]),
        ], axis=1)

        self.mbox_conf_reahpe = F.reshape(self.mbox_conf, (7308 * batchsize, 21))
        self.mbox_conf_softmax = F.softmax(self.mbox_conf_reahpe)
        self.mbox_conf_softmax_reahpe = F.reshape(self.mbox_conf, (batchsize, 7308, 21))
        if self.train:
            self.loss = self.loss_func(h, t)
            self.accuracy = self.loss
            return self.loss

    def prior(self, size, min_size, max_size, aspect, flip, clip, variance):
        aspect_ratio = [1.]
        for i in aspect:
            aspect_ratio.append(i)
            aspect_ratio.append(1/i)
        height, width = size
        img_width = img_height = 300.
        step_x = img_width / float(width)
        step_y = img_width / float(height)
        top_data = np.zeros([height, width, (len(aspect_ratio) + bool(max_size)), 2, 4])

        wid_hei_list = [(min_size, min_size)]
        if max_size > 0:
            #second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
            wid_hei_list.append((np.sqrt(min_size * max_size), np.sqrt(min_size * max_size)))

        for ar in aspect_ratio:
            if abs(ar - 1.) < 1e-6:
                continue
            box_width = min_size * np.sqrt(ar)
            box_height = min_size / np.sqrt(ar)
            wid_hei_list.append((box_width, box_height))

        for h in range(height):
            for w in range(width):
                center_x = (w + 0.5) * step_x
                center_y = (h + 0.5) * step_y
                for idx, w_h in enumerate(wid_hei_list):
                    box_width, box_height = w_h
                    top_data[h, w, idx, 0, 0] = (center_x - box_width / 2.) / img_width
                    top_data[h, w, idx, 0, 1] = (center_y - box_height / 2.) / img_height
                    top_data[h, w, idx, 0, 2] = (center_x + box_width / 2.) / img_width
                    top_data[h, w, idx, 0, 3] = (center_y + box_height / 2.) / img_height
        if clip:
            top_data[np.where(top_data < 0)] = 0
            top_data[np.where(top_data > 1)] = 1
        top_data[:, :, :, 1] = variance
        return top_data

    def detection(self, idx, nms_th=0.45, cls_th=0.6):
        prior = self.mbox_prior
        loc = self.mbox_loc.data[idx]
        conf = self.mbox_conf_softmax.data
        cand = []
        for label in range(1, 21):
            label_cand = np.array([np.hstack([label, score, self.decoder(loc[pos], prior[pos])]) for pos, score in enumerate(conf[:, label]) if score > 0.1])
            if label_cand.any():
                k = self.nms(label_cand[:, 2:], label_cand[:, 1], 0.1, nms_th, 200)
                for i in k:
                    cand.append(label_cand[i])
        cand = np.array(cand)
        cand = cand[np.where(cand[:, 1] >= cls_th)]
        return cand

    def decoder(self, loc, prior):
        prior_data = prior[1]
        prior = prior[0]
        bbox_data = np.array([0] * 4, dtype=np.float32)
        p_xmin, p_ymin, p_xmax, p_ymax = prior
        xmin, ymin, xmax, ymax = loc
        prior_width = p_xmax - p_xmin
        prior_height = p_ymax - p_ymin
        prior_center_x = (p_xmin + p_xmax) / 2.
        prior_center_y = (p_ymin + p_ymax) / 2.
        decode_bbox_center_x = prior_data[0] * xmin * prior_width + prior_center_x;
        decode_bbox_center_y = prior_data[1] * ymin * prior_height + prior_center_y;
        decode_bbox_width = np.exp(prior_data[2] * xmax) * prior_width;
        decode_bbox_height = np.exp(prior_data[3] * ymax) * prior_height;
        bbox_data[0] = decode_bbox_center_x - decode_bbox_width / 2.
        bbox_data[1] = decode_bbox_center_y - decode_bbox_height / 2.
        bbox_data[2] = decode_bbox_center_x + decode_bbox_width / 2.
        bbox_data[3] = decode_bbox_center_y + decode_bbox_height / 2.
        return bbox_data

    def encoder(self, prior_bbox, bbox, prior_variance):
        encode_bbox = np.array([0]*4, dtype=np.float32)
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

    def nms(self, bboxes, scores, score_th, nms_th, top_k):
        score_iter = 0
        score_index = scores.argsort()[::-1][:top_k]
        indices = []
        while(score_iter < len(score_index)):
            idx = score_index[score_iter]
            keep = True
            cand_bbox=bboxes[idx]
            if cand_bbox[0] == cand_bbox[2] or cand_bbox[1] == cand_bbox[3]:
                keep=False
            for i in range(len(indices)):
                if keep:
                    kept_idx = indices[i]
                    overlap = self.IoU(bboxes[idx], bboxes[kept_idx])
                    keep = overlap <= nms_th
                else:
                    break
            if keep:
                indices.append(idx)
            score_iter+=1
        return indices

    def IoU(self, a, b):
        I = self.intersection(a, b)
        if not I:
            return 0
        a_ = (a[2]-a[0])*(a[3]-a[1])
        b_ = (b[2]-b[0])*(b[3]-b[1])
        if a_ <=0 or b_ <= 0:
            return 1
        i = (I[2]-I[0])*(I[3]-I[1])
        return i/(a_ + b_ - i*2)

    def intersection(self, a,b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        w = x2 - x1
        h = y2 - y1
        if w<0 or h<0: return ()
        return (x1, y1, x2, y2)
