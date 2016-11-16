import argparse

import numpy as np
#import cv2
import skimage.io
import skimage.draw
from skimage.transform import resize
import chainer
from chainer import serializers


import ssd_net


parser = argparse.ArgumentParser(
    description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('path', help='Path to training image-label list file')
args = parser.parse_args()
mean = np.array([104,117,123])
img = skimage.img_as_float(skimage.io.imread(args.path, as_grey=False)).astype(np.float32)
moto = img
img = resize(img, (300,300))
img = img*255 - mean[::-1]
img = img.transpose(2, 0, 1)[::-1]

model = ssd_net.SSD()
serializers.load_npz("ssd.model", model)
x = chainer.Variable(np.array([img],dtype=np.float32))
model(x,1)

def nms(bboxes, scores, score_th, nms_th, top_k):
    score_iter = 0
    score_index = scores.argsort()[::-1][:top_k]
    indices = []
    while(score_iter < len(score_index)):
        idx = score_index[score_iter]
        keep = True
        print(idx)
        for i in range(len(indices)):
            if keep:
                kept_idx = indices[i]
                print(bboxes[idx], bboxes[kept_idx])
                overlap = IoU(bboxes[idx], bboxes[kept_idx])
                keep = overlap <= nms_th
            else:
                break
        if keep:
            indices.append(idx)
        score_iter+=1
    return indices
"""
def nms(bboxes, nms_th, top_k):
    bboxes = bboxes[:top_k]
    indices = []
    for bbox in bboxes:
        keep = True
        for i in range(indices):
            if keep:
                kept_idx = indices[i]
                overlap = IoU(bbox, bboxes[kept_idx])
                keep = overlap <= nms_th
            else:
                break
        if keep:
            indices.append(idx)
    return indices
"""
def IoU(a, b):
    #U = union(a, b)
    I = intersection(a, b)
    if not I:
        return 0
    a_ = (a[2]-a[0])*(a[3]-a[1])
    b_ = (b[2]-b[0])*(b[3]-b[1])
    if a_ <=0 or b_ <= 0:
        return 1
    i = (I[2]-I[0])*(I[3]-I[1])
    return i/(a_ + b_ - i*2)

def intersection(a,b):
  x1 = max(a[0], b[0])
  y1 = max(a[1], b[1])
  x2 = min(a[2], b[2])
  y2 = min(a[3], b[3])
  w = x2 - x1
  h = y2 - y1
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x1, y1, x2, y2)

a=model.detection()
for i in a:
    conf, x1, y1, x2, y2 = i
    x1 *= moto.shape[1]
    x2 *= moto.shape[1]
    y1 *=moto.shape[0]
    y2 *=moto.shape[0]
    rr,cc = skimage.draw.polygon_perimeter([y1, y2, y2, y1],[x1, x1, x2, x2], shape=moto.shape, clip=True)
    moto[rr, cc]= 0
