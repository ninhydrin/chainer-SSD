import argparse

import numpy as np
import cv2
import skimage.io
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
img = resize(img, (300,300))
img = img*255 - mean[::-1]
img = img.transpose(2, 0, 1)[::-1]

model = ssd_net.SSD()
serializers.load_npz("ssd.model", model)
x = chainer.Variable(np.array([img],dtype=np.float32))
model(x,1)

def nms(bboxes, scores, score_th, nms_th, top_k):
    score_iter = 0
    score_index = scores.argsort[::-1][:top_k]
    indices = []
    while(score_iter < len(score_index)):
        idx = score_index[score_iter]
        keep = True
        for i in range(indices):
            if keep:
                kept_idx = indices[i]
                overlap = IoU(bboxes[idx], bboxes[kept_idx])
                keep = overlap <= nms_th
            else:
                break
        if keep:
            indices.append(idx)
        score_iter+=1
    return indices

def IoU(a, b):
    U = union(a, b)
    I = intersection(a, b)

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)
