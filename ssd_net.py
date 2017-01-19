# encoding:utf-8
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
import numpy as np

from util.prior import prior


#xp = cuda.cupy

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


            conv4_3_norm_mbox_loc=L.Convolution2D(
                512, 12,  3, pad=1),  # 3 prior boxes
            conv4_3_norm_mbox_conf=L.Convolution2D(512, 63,  3, pad=1),

            fc7_mbox_loc=L.Convolution2D(1024, 24, 3, pad=1),  # 6 prior boxes
            fc7_mbox_conf=L.Convolution2D(1024, 126, 3, pad=1),

            conv6_2_mbox_loc=L.Convolution2D(
                512, 24, 3, pad=1),  # 6 prior boxes
            conv6_2_mbox_conf=L.Convolution2D(512, 126, 3, pad=1),

            conv7_2_mbox_loc=L.Convolution2D(
                256, 24, 3, pad=1),  # 6 prior boxes
            conv7_2_mbox_conf=L.Convolution2D(256, 126, 3, pad=1),

            conv8_2_mbox_loc=L.Convolution2D(
                256, 24, 3, pad=1),  # 6 prior boxes
            conv8_2_mbox_conf=L.Convolution2D(256, 126, 3, pad=1),

            pool6_mbox_loc=L.Convolution2D(256, 24, 3, pad=1),
            pool6_mbox_conf=L.Convolution2D(
                256, 126, 3, pad=1),  # 6 prior boxes

        )
        self.train = False
        self.set_info("c4", 38, 38, 3)
        self.set_info("f7", 19, 19, 6)
        self.set_info("c6", 10, 10, 6)
        self.set_info("c7", 5, 5, 6)
        self.set_info("c8", 3, 3, 6)
        self.set_info("p6", 1, 1, 6)

        self.conv4_3_norm_mbox_priorbox = prior(
            (38, 38), 30., 0, [2], 1, 1, (0.1, 0.1, 0.2, 0.2))
        self.fc7_mbox_priorbox = prior(
            (19, 19), 60., 114., [2, 3], 1, 1, (0.1, 0.1, 0.2, 0.2))
        self.conv6_2_mbox_priorbox = prior(
            (10, 10), 114., 168., [2, 3], 1, 1, (0.1, 0.1, 0.2, 0.2))
        self.conv7_2_mbox_priorbox = prior(
            (5, 5), 168., 222., [2, 3], 1, 1, (0.1, 0.1, 0.2, 0.2))
        self.conv8_2_mbox_priorbox = prior(
            (3, 3), 222., 276., [2, 3], 1, 1, (0.1, 0.1, 0.2, 0.2))
        self.pool6_mbox_priorbox = prior(
            (1, 1), 276., 330., [2, 3], 1, 1, (0.1, 0.1, 0.2, 0.2))
        self.mbox_prior = np.concatenate([
            self.conv4_3_norm_mbox_priorbox.reshape(-1, 2, 4),
            self.fc7_mbox_priorbox.reshape(-1, 2, 4),
            self.conv6_2_mbox_priorbox.reshape(-1, 2, 4),
            self.conv7_2_mbox_priorbox.reshape(-1, 2, 4),
            self.conv8_2_mbox_priorbox.reshape(-1, 2, 4),
            self.pool6_mbox_priorbox.reshape(-1, 2, 4),
        ], axis=0)

    def __call__(self, x, conf, loc, conf_mask, loc_mask):
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
        kari = F.reshape(self.h_conv4_3, (batchsize * ch, hh * ww))
        kari = F.transpose(kari, (1, 0))
        kari = F.normalize(kari)
        kari = F.transpose(kari, (1, 0))
        kari = F.reshape(kari, (batchsize, ch, hh, ww))

        self.h_conv4_3_norm = self.normalize(kari)
        self.h_conv4_3_norm_mbox_loc = self.conv4_3_norm_mbox_loc(
            self.h_conv4_3_norm)
        self.h_conv4_3_norm_mbox_conf = self.conv4_3_norm_mbox_conf(
            self.h_conv4_3_norm)
        self.h_conv4_3_norm_mbox_loc_perm = F.transpose(
            self.h_conv4_3_norm_mbox_loc, (0, 2, 3, 1))
        self.h_conv4_3_norm_mbox_conf_perm = F.transpose(
            self.h_conv4_3_norm_mbox_conf, (0, 2, 3, 1))

        self.h_conv4_3_norm_mbox_loc_flat = F.reshape(
            self.h_conv4_3_norm_mbox_loc_perm, (batchsize, self.c4_h, self.c4_w, self.c4_d, 4))
        self.h_conv4_3_norm_mbox_conf_flat = F.reshape(
            self.h_conv4_3_norm_mbox_conf_perm, (batchsize, self.c4_h, self.c4_w, self.c4_d, 21))

        self.h_fc7_mbox_loc = self.fc7_mbox_loc(self.h_fc7)
        self.h_fc7_mbox_conf = self.fc7_mbox_conf(self.h_fc7)
        self.h_fc7_mbox_loc_perm = F.transpose(
            self.h_fc7_mbox_loc, (0, 2, 3, 1))
        self.h_fc7_mbox_conf_perm = F.transpose(
            self.h_fc7_mbox_conf, (0, 2, 3, 1))
        self.h_fc7_mbox_loc_flat = F.reshape(
            self.h_fc7_mbox_loc_perm, (batchsize, self.f7_h, self.f7_w, self.f7_d, 4))
        self.h_fc7_mbox_conf_flat = F.reshape(
            self.h_fc7_mbox_conf_perm, (batchsize, self.f7_h, self.f7_w, self.f7_d, 21))

        self.h_conv6_2_mbox_loc = self.conv6_2_mbox_loc(self.h_conv6_2)
        self.h_conv6_2_mbox_conf = self.conv6_2_mbox_conf(self.h_conv6_2)
        self.h_conv6_2_mbox_loc_perm = F.transpose(
            self.h_conv6_2_mbox_loc, (0, 2, 3, 1))
        self.h_conv6_2_mbox_conf_perm = F.transpose(
            self.h_conv6_2_mbox_conf, (0, 2, 3, 1))
        self.h_conv6_2_mbox_loc_flat = F.reshape(
            self.h_conv6_2_mbox_loc_perm, (batchsize, self.c6_h, self.c6_w, self.c6_d, 4))
        self.h_conv6_2_mbox_conf_flat = F.reshape(
            self.h_conv6_2_mbox_conf_perm, (batchsize, self.c6_h, self.c6_w, self.c6_d, 21))

        self.h_conv7_2_mbox_loc = self.conv7_2_mbox_loc(self.h_conv7_2)
        self.h_conv7_2_mbox_conf = self.conv7_2_mbox_conf(self.h_conv7_2)
        self.h_conv7_2_mbox_loc_perm = F.transpose(
            self.h_conv7_2_mbox_loc, (0, 2, 3, 1))
        self.h_conv7_2_mbox_conf_perm = F.transpose(
            self.h_conv7_2_mbox_conf, (0, 2, 3, 1))
        self.h_conv7_2_mbox_loc_flat = F.reshape(
            self.h_conv7_2_mbox_loc_perm, (batchsize, self.c7_h, self.c7_w, self.c7_d, 4))
        self.h_conv7_2_mbox_conf_flat = F.reshape(
            self.h_conv7_2_mbox_conf_perm, (batchsize, self.c7_h, self.c7_w, self.c7_d, 21))

        self.h_conv8_2_mbox_loc = self.conv8_2_mbox_loc(self.h_conv8_2)
        self.h_conv8_2_mbox_conf = self.conv8_2_mbox_conf(self.h_conv8_2)
        self.h_conv8_2_mbox_loc_perm = F.transpose(
            self.h_conv8_2_mbox_loc, (0, 2, 3, 1))
        self.h_conv8_2_mbox_conf_perm = F.transpose(
            self.h_conv8_2_mbox_conf, (0, 2, 3, 1))
        self.h_conv8_2_mbox_loc_flat = F.reshape(
            self.h_conv8_2_mbox_loc_perm, (batchsize, self.c8_h, self.c8_w, self.c8_d, 4))
        self.h_conv8_2_mbox_conf_flat = F.reshape(
            self.h_conv8_2_mbox_conf_perm, (batchsize, self.c8_h, self.c8_w, self.c8_d, 21))

        self.h_pool6_mbox_loc = self.pool6_mbox_loc(self.h_pool6)
        self.h_pool6_mbox_conf = self.pool6_mbox_conf(self.h_pool6)
        self.h_pool6_mbox_loc_perm = F.transpose(
            self.h_pool6_mbox_loc, (0, 2, 3, 1))
        self.h_pool6_mbox_conf_perm = F.transpose(
            self.h_pool6_mbox_conf, (0, 2, 3, 1))
        self.h_pool6_mbox_loc_flat = F.reshape(
            self.h_pool6_mbox_loc_perm, (batchsize, self.p6_h, self.p6_w, self.p6_d, 4))
        self.h_pool6_mbox_conf_flat = F.reshape(
            self.h_pool6_mbox_conf_perm, (batchsize, self.p6_h, self.p6_w, self.p6_d, 21))

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

        self.mbox_conf_reahpe = F.reshape(
            self.mbox_conf, (7308 * batchsize, 21))
        self.mbox_conf_softmax = F.softmax(self.mbox_conf_reahpe)
        self.mbox_conf_softmax_reahpe = F.reshape(
            self.mbox_conf_softmax, (batchsize, 7308, 21))

        if self.train:
            mbox_conf = cuda.to_cpu(self.mbox_conf_softmax_reahpe.data)
            dammy_label = np.zeros([batchsize, 7308, 21])
            for i in range(batchsize):
                self.conf_num = int(conf_mask[i].sum())
                self.mask = conf_mask[i].copy()
                negative_sample_num = int(conf_mask[i].sum() * 5) if  int(conf_mask[i].sum() * 5) < 4000 else 4000
                self.num = negative_sample_num
                negative_index = mbox_conf[i, :, 0].argsort()[: negative_sample_num]
                self.ind = negative_index
                self.conf_mask = conf_mask[i]
                conf_mask[i, negative_index] = 1
                dammy_label[i][np.where(conf_mask[i, :, 0] == 0)][0] = 100
            t_conf_mask = chainer.Variable(cuda.cupy.array(conf_mask), volatile=x.volatile)
            t_loc_mask = chainer.Variable(cuda.cupy.array(loc_mask), volatile=x.volatile)
            dammy_label = cuda.cupy.array(dammy_label)
            self.t_conf_mask=t_conf_mask
            train_conf = self.mbox_conf * t_conf_mask
            train_conf.data += dammy_label
            self.train_conf = F.reshape(train_conf, (-1, 21))
            #train_conf = F.reshape(self.mbox_conf * t_conf_mask, (-1, 21))
            #print(type(dammy_label), type(self.train_conf.data))

            self.val_conf = F.flatten(conf)
            self.loss = F.softmax_cross_entropy(self.train_conf, self.val_conf)
            self.loss += F.mean_squared_error(self.mbox_loc * t_loc_mask, loc)
            self.accuracy = F.accuracy(self.train_conf, self.val_conf)
            return self.loss
        else:
            return self.mbox_loc, self.mbox_conf_softmax_reahpe
