# encoding:utf-8
import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

class SSD (chainer.Chain):
    insize = 300
    def __init__(self):
        super(SSD, self).__init__(
            conv1_1 =  L.Convolution2D(3,  64, 3, pad=1),
            conv1_2 =  L.Convolution2D(64,  64, 3, pad=1),
            conv2_1 = L.Convolution2D(64, 128,  3, pad=1),
            conv2_2 = L.Convolution2D(128, 128,  3, pad=1),
            conv3_1= L.Convolution2D(128, 256,  3, pad=1),
            conv3_2= L.Convolution2D(256, 256,  3, pad=1),
            conv3_3= L.Convolution2D(256, 256,  3, pad=1),

            conv4_1= L.Convolution2D(256, 512,  3, pad=1),
            conv4_2= L.Convolution2D(512, 512,  3, pad=1),
            conv4_3= L.Convolution2D(512, 512,  3, pad=1),

            conv5_1= L.Convolution2D(512, 512,  3, pad=1),
            conv5_2= L.Convolution2D(512, 512,  3, pad=1),
            conv5_3= L.Convolution2D(512, 512,  3, pad=1),

            #fc6 = L.Convolution2D(512, 1024,  3, pad=6),
            fc6 = L.DilatedConvolution2D(512, 1024,  3, pad=6,dilate=6),
            fc7 = L.Convolution2D(1024, 1024,  1),

            conv6_1 = L.Convolution2D(1024, 256,  1),
            conv6_2 = L.Convolution2D(256, 512,  3, stride=2, pad=1),

            conv7_1 = L.Convolution2D(512, 128,  1),
            conv7_2 = L.Convolution2D(128, 256,  3, stride=2, pad=1),

            conv8_1 = L.Convolution2D(256, 128,  1),
            conv8_2 = L.Convolution2D(128, 256,  3, stride=2, pad=1),

            normalize = L.Scale(W_shape=512),


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
        self.conv4_3_norm_priorbox = self.prior((38, 38), 30., 0, [2], 1, 1,(0.1, 0.1, 0.2, 0.2))

    def __call__(self, x, t):

        h = F.relu(self.conv1_1(x))
        h = F.max_pooling_2d(F.relu(self.conv1_2(h)),2,2)
        h = F.relu(self.conv2_1(h))
        h = F.max_pooling_2d(F.relu(self.conv2_2(h)),2,2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.max_pooling_2d(F.relu(self.conv3_3(h)),2,2)
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

        batchsize,ch,hh,ww = self.h_conv4_3.shape
        kari = F.reshape(self.h_conv4_3,(batchsize*ch, hh*ww))
        kari = F.transpose(kari, (1,0))
        kari = F.normalize(kari)
        kari = F.transpose(kari, (1,0))
        kari = F.reshape(kari,(batchsize, ch, hh, ww))

        self.h_conv4_3_norm = self.normalize(kari)
        self.h_conv4_3_norm_mbox_loc = self.conv4_3_norm_mbox_loc(self.h_conv4_3_norm)
        self.h_conv4_3_norm_mbox_conf = self.conv4_3_norm_mbox_conf(self.h_conv4_3_norm)
        self.h_conv4_3_norm_mbox_loc_perm = F.transpose(self.h_conv4_3_norm_mbox_loc,(0,2,3,1))
        self.h_conv4_3_norm_mbox_conf_perm = F.transpose(self.h_conv4_3_norm_mbox_conf,(0,2,3,1))
        self.h_conv4_3_norm_mbox_loc_flat = F.flatten(self.h_conv4_3_norm_mbox_loc_perm)
        self.h_conv4_3_norm_mbox_conf_flat = F.flatten(self.h_conv4_3_norm_mbox_conf_perm)

        self.h_fc7_mbox_loc = self.fc7_mbox_loc(self.h_fc7)
        self.h_fc7_mbox_conf = self.fc7_mbox_conf(self.h_fc7)
        self.h_fc7_mbox_loc_perm = F.transpose(self.h_fc7_mbox_loc,(0,2,3,1))
        self.h_fc7_mbox_conf_perm = F.transpose(self.h_fc7_mbox_conf,(0,2,3,1))
        self.h_fc7_mbox_loc_flat = F.flatten(self.h_fc7_mbox_loc_perm)
        self.h_fc7_mbox_conf_flat = F.flatten(self.h_fc7_mbox_conf_perm)

        self.h_conv6_2_mbox_loc = self.conv6_2_mbox_loc(self.h_conv6_2)
        self.h_conv6_2_mbox_conf = self.conv6_2_mbox_conf(self.h_conv6_2)
        self.h_conv6_2_mbox_loc_perm = F.transpose(self.h_conv6_2_mbox_loc,(0,2,3,1))
        self.h_conv6_2_mbox_conf_perm = F.transpose(self.h_conv6_2_mbox_conf,(0,2,3,1))
        self.h_conv6_2_mbox_loc_flat = F.flatten(self.h_conv6_2_mbox_loc_perm)
        self.h_conv6_2_mbox_conf_flat = F.flatten(self.h_conv6_2_mbox_conf_perm)

        self.h_conv7_2_mbox_loc = self.conv7_2_mbox_loc(self.h_conv7_2)
        self.h_conv7_2_mbox_conf = self.conv7_2_mbox_conf(self.h_conv7_2)
        self.h_conv7_2_mbox_loc_perm = F.transpose(self.h_conv7_2_mbox_loc,(0,2,3,1))
        self.h_conv7_2_mbox_conf_perm = F.transpose(self.h_conv7_2_mbox_conf,(0,2,3,1))
        self.h_conv7_2_mbox_loc_flat = F.flatten(self.h_conv7_2_mbox_loc_perm)
        self.h_conv7_2_mbox_conf_flat = F.flatten(self.h_conv7_2_mbox_conf_perm)

        self.h_conv8_2_mbox_loc = self.conv8_2_mbox_loc(self.h_conv8_2)
        self.h_conv8_2_mbox_conf = self.conv8_2_mbox_conf(self.h_conv8_2)
        self.h_conv8_2_mbox_loc_perm = F.transpose(self.h_conv8_2_mbox_loc,(0,2,3,1))
        self.h_conv8_2_mbox_conf_perm = F.transpose(self.h_conv8_2_mbox_conf,(0,2,3,1))
        self.h_conv8_2_mbox_loc_flat = F.flatten(self.h_conv8_2_mbox_loc_perm)
        self.h_conv8_2_mbox_conf_flat = F.flatten(self.h_conv8_2_mbox_conf_perm)

        self.h_pool6_mbox_loc = self.pool6_mbox_loc(self.h_pool6)
        self.h_pool6_mbox_conf = self.pool6_mbox_conf(self.h_pool6)
        self.h_pool6_mbox_loc_perm = F.transpose(self.h_pool6_mbox_loc,(0,2,3,1))
        self.h_pool6_mbox_conf_perm = F.transpose(self.h_pool6_mbox_conf,(0,2,3,1))
        self.h_pool6_mbox_loc_flat = F.flatten(self.h_pool6_mbox_loc_perm)
        self.h_pool6_mbox_conf_flat = F.flatten(self.h_pool6_mbox_conf_perm)

        #TODO バッチサイズの考慮
        self.mbox_loc = F.concat([self.h_conv4_3_norm_mbox_loc_flat,
                                  self.h_fc7_mbox_loc_flat,
                                  self.h_conv6_2_mbox_loc_flat,
                                  self.h_conv7_2_mbox_loc_flat,
                                  self.h_conv8_2_mbox_loc_flat,
                                  self.h_pool6_mbox_loc_flat],axis=0)

        self.mbox_conf = F.concat([self.h_conv4_3_norm_mbox_conf_flat,
                                   self.h_fc7_mbox_conf_flat,
                                   self.h_conv6_2_mbox_conf_flat,
                                   self.h_conv7_2_mbox_conf_flat,
                                   self.h_conv8_2_mbox_conf_flat,
                                   self.h_pool6_mbox_conf_flat],axis=0)

        self.mbox_reahpe = F.reshape(self.mbox_conf,(1,7308,21))
        self.mbox_reahpe_softmax = F.softmax(self.mbox_reahpe)
        self.loss = self.loss_func(h, t)
        self.accuracy = self.loss
        return self.loss

    def prior(self, h, min_size, max_size,aspect, flip, clip, variance):
        aspect_ratio = [1.]
        for i in aspect:
            aspect_ratio.append(i)
            aspect_ratio.append(1/i)
        height, width = h
        img_width = img_height = 300.
        step_x = img_width / float(width)
        step_y = img_width / float(height)
        top_data=np.zeros(height * width * len(aspect_ratio) * 4)
        idx=0
        for h in range(height):
            for w in range(width):
                center_x = (w + 0.5) * step_x
                center_y = (h + 0.5) * step_y
                box_width = box_height = min_size
                top_data[idx] = (center_x - box_width / 2.) / img_width
                idx+=1
                top_data[idx] = (center_y - box_height / 2.) / img_height
                idx+=1
                top_data[idx] = (center_x + box_width / 2.) / img_width
                idx+=1
                top_data[idx] = (center_y + box_height / 2.) / img_height
                idx+=1
                if max_size > 0:
                    #second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                    box_width = box_height = np.sqrt(min_size * max_size)
                    top_data[idx] = (center_x - box_width / 2.) / img_width
                    idx+=1
                    top_data[idx] = (center_y - box_height / 2.) / img_height
                    idx+=1
                    top_data[idx] = (center_x + box_width / 2.) / img_width
                    idx+=1
                    top_data[idx] = (center_y + box_height / 2.) / img_height
                    idx+=1
                for ar in aspect_ratio:
                    if abs(ar - 1.) < 1e-6:
                        continue
                    box_width = min_size * np.sqrt(ar)
                    box_height = min_size / np.sqrt(ar)

                    top_data[idx] = (center_x - box_width / 2.) / img_width
                    idx+=1
                    top_data[idx] = (center_y - box_height / 2.) / img_height
                    idx+=1
                    top_data[idx] = (center_x + box_width / 2.) / img_width
                    idx+=1
                    top_data[idx] = (center_y + box_height / 2.) / img_height
                    idx+=1

        if clip:
            for i in range(len(top_data)):
                if top_data[i] > 1:
                    top_data[i] = 1
                elif top_data[i] < 0:
                    top_data[i] = 0
        val_data=np.zeros(height * width * len(aspect_ratio) * 4)
        if len(variance)==1:
            pass
        else:
            count = 0
            for h in range(height):
                for w in range(width):
                    for ar in range(len(aspect_ratio)):
                        for v in variance:
                            val_data[count] = v
                            count+=1
        return np.vstack([top_data, val_data])

    def decoder(prior, loc, prior_data):
        bbox_data = np.array([0]*4,dtype=np.float32)
        p_xmin, p_ymin, p_xmax, p_ymax= prior
        xmin, ymin, xmax, ymax= loc
        prior_width = p_xmax - p_xmin
        prior_height = p_ymax - p_ymin
        prior_center_x = (p_xmin + p_xmax) / 2.
        prior_center_y = (p_ymin + p_ymax) / 2.
        decode_bbox_center_x = prior_data[0] * xmin * prior_width + prior_center_x;
        decode_bbox_center_y = prior_data[1] * ymin * prior_height + prior_center_y;
        decode_bbox_width = np.exp(prior_data[2] * xmax) * prior_width;
        decode_bbox_height = np.exp(prior_data[3] * ymax) * prior_height;
        bbox_data[0] = decode_bbox_center_x - decode_bbox_width / 2.
        bbox_data[1] = decode_bbox_center_y - decode_bbox_height / 2.;
        bbox_data[2] = decode_bbox_center_x + decode_bbox_width / 2.;
        bbox_data[3] = decode_bbox_center_y + decode_bbox_height / 2.;
        return bbox_data
