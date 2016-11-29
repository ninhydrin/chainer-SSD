import argparse

import matplotlib.pyplot as plt
import numpy as np
#import cv2
import skimage.io
import skimage.draw
from skimage.transform import resize
import chainer
from chainer import serializers


import ssd_net

labelmap = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

parser = argparse.ArgumentParser(
    description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('path', help='Path to training image-label list file')
args = parser.parse_args()
mean = np.array([104,117,123])
image = skimage.img_as_float(skimage.io.imread(args.path, as_grey=False)).astype(np.float32)

img = resize(image, (300,300))
img = img*255 - mean[::-1]
img = img.transpose(2, 0, 1)[::-1]

model = ssd_net.SSD()
serializers.load_npz("ssd.model", model)
x = chainer.Variable(np.array([img],dtype=np.float32))
model(x,1)
"""
a=model.detection()
plt.imshow(image)
currentAxis = plt.gca()
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
for i in a:
    label, conf, x1, y1, x2, y2 = i
    label = int(label) -1
    x1 = int(round(x1 * image.shape[1]))
    x2 = int(round(x2 * image.shape[1]))
    y1 = int(round(y1 * image.shape[0]))
    y2 = int(round(y2 * image.shape[0]))
    label_name = labelmap[int(label)]
    display_txt = '%s: %.2f'%(label_name, conf)
    coords = (x1, y1), x2-x1+1, y2-y1+1
    color = colors[int(label)]
    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    currentAxis.text(x1, y1, display_txt, bbox={'facecolor':color, 'alpha':0.5})
#plt.show()
"""
