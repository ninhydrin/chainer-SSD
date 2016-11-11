import chainer
import chainer.links as L
import chainer.functions as F

class SSD (chainer.Chain):
    insize = 300
    def __init__(self):
        super(SSD, self).__init__(
            conv1_1 =  L.Convolution2D(3,  64, 7, pad=1),
            conv1_2 =  L.Convolution2D(3,  64, 7, pad=1),
            conv2_1 = L.Convolution2D(64, 128,  3, pad=1),
            conv2_2 = L.Convolution2D(128, 192,  3, pad=1),
            conv3_1= L.Convolution2D(192, 256,  3, pad=1),
            conv3_2= L.Convolution2D(128, 256,  3, pad=1),
            conv3_3= L.Convolution2D(256, 256,  3, pad=1),

            conv4_1= L.Convolution2D(256, 512,  3, pad=1),
            conv4_2= L.Convolution2D(512, 512,  3, pad=1),
            conv4_3= L.Convolution2D(512, 512,  3, pad=1),

            conv5_1= L.Convolution2D(512, 512,  3, pad=1),
            conv5_2= L.Convolution2D(512, 512,  3, pad=1),
            conv5_3= L.Convolution2D(512, 512,  3, pad=1),

            fc6 = L.Convolution2D(512, 1024,  3, pad=6),
            fc7 = L.Convolution2D(1024, 1024,  1),

            conv6_1 = L.Convolution2D(1024, 256,  1),
            conv6_2 = L.Convolution2D(256, 512,  3, stride=2, pad=1),

            conv7_1 = L.Convolution2D(512, 128,  1),
            conv7_2 = L.Convolution2D(128, 256,  3, stride=2, pad=1),

            conv8_1 = L.Convolution2D(256, 128,  1),
            conv8_2 = L.Convolution2D(128, 256,  3, stride=2, pad=1),

            normalize = L.Scale(512),

            conv4_3_norm_mbox_loc = L.Convolution2D(512, 12,  3, pad=1), #3 prior boxes
            conv4_3_norm_mbox_conf = L.Convolution2D(512, 64,  3, pad=1),

            fc7_mbox_loc = L.Convolution2D(1024, 24, 3, pad=1), #6 prior boxes
            fc7_mbox_conf = L.Convolution2D(1024, 126, 3, pad=1),

            conv6_2_mbox_loc = L.Convolution2D(512, 24, 3, pad=1), #6 prior boxes
            conv6_2_mbox_conf = L.Convolution2D(512, 126, 3, pad=1),

            conv7_2_mbox_loc = L.Convolution2D(256, 24, 3, pad=1), #6 prior boxes
            conv7_2_mbox_conf = L.Convolution2D(256, 126, 3, pad=1),

            conv8_2_mbox_loc = L.Convolution2D(256, 24, 3, pad=1), #6 prior boxes
            conv8_2_mbox_conf = L.Convolution2D(256, 126, 3, pad=1),

            pool6_mbox_loc = L.Convolution2D(128, 24, 3, pad=1),
            pool6_mbox_conf = L.Convolution2D(128, 126, 3, pad=1), #6 prior boxes

        )
        self.train = False

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
        h = F.max_pooling_2d(F.relu(self.conv4_3(h)),2,2)

        self.h_conv4_3 = h
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.max_pooling_2d(F.relu(self.conv5_3(h)), 3, 1)
        h = F.relu(self.fc6(h))
        self.h_fc6 = h
        h = F.relu(self.fc7(h))

        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h))
        self.h_conv6_2 = h

        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h))
        self.h_conv7_2 = h

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        self.h_conv8_2 = h

        h = F.average_pooling_2d(h)

        self.h_pool = h
        self.loss = self.loss_func(h, t)
        self.accuracy = self.loss
        return self.loss
