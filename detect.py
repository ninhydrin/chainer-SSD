import os
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.draw
from skimage.transform import resize
import chainer
from chainer import serializers

import ssd_net
import draw

labelmap = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

parser = argparse.ArgumentParser(
    description='detect object')
parser.add_argument('path', help='Path to image file list')
parser.add_argument('--cls_th', default=0.6, help='Threshold of confidence')
parser.add_argument('--nms_th', default=0.5, help='Threshold of nms')
parser.add_argument('--result', default="reslut", help='Path to result dir')
parser.add_argument('--year', default="2007", help='Year of eval')
args = parser.parse_args()

if not os.path.isdir(args.result):
    os.mkdir(args.result)

for label in labelmap:
    with open("{}/{}.txt".format(args.result, label), "w") as f:
        pass
mean = np.array([104, 117, 123])
model = ssd_net.SSD()
serializers.load_npz("ssd.model", model)
prior = model.mbox_prior.astype(np.float32)
template = "{} {} {} {} {} {}\n"
path_temp = "VOCdevkit/VOC{}/JPEGImages/{}.jpg"
with open(args.path) as img_list:
    for num in img_list: 
        num = num.strip()
        sys.stdout.write("\r{}".format(num))
        sys.stdout.flush()
        path = path_temp.format(args.year, num)
        image = skimage.img_as_float(skimage.io.imread(path, as_grey=False)).astype(np.float32)
        img = resize(image, (300, 300))
        img = img*255 - mean[::-1]
        img = img.transpose(2, 0, 1)[::-1]

        x = chainer.Variable(np.array([img], dtype=np.float32))
        model(x, 1)

        loc = model.mbox_loc.data[0]
        conf = model.mbox_conf_softmax_reahpe.data[0]
        cand = draw.detect(prior, loc, conf, nms_th=args.nms_th, cls_th=args.cls_th)
        for i in cand:
            label, conf, x1, y1, x2, y2 = i
            label = int(label) - 1
            label_name = labelmap[int(label)]
            x1 = int(round(x1 * image.shape[1]))
            x2 = int(round(x2 * image.shape[1]))
            y1 = int(round(y1 * image.shape[0]))
            y2 = int(round(y2 * image.shape[0]))
            with open("{}/{}.txt".format(args.result, label_name), "a") as f:
                f.write(template.format(num, conf, x1, y1, x2, y2))
