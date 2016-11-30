import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import xml.dom.minidom

import numpy as np

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert_annotation(path):
    tree=ET.parse(open(path))
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    data = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        bb = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]
        data.append([cls_id]+bb)
    return (h, w), np.array(data)

def make_data(year):
    data_type = ("trainval","test")
    voc_path = "VOCdevkit/VOC{}/".format(year)
    re = []
    for i in data_type:
        data = []
        name_list = [j[:-1] for j in open(voc_path + "ImageSets/Main/{}.txt".format(i))]
        f_name = "{}_voc{}.pkl".format(i, year)
        for img_name in name_list:
            path = voc_path + "Annotations/{}.xml".format(img_name)
            size, obj =  convert_annotation(path)
            data.append([path, size, obj])
        pickle.dump(data, open(f_name, "wb"))

if __name__ == "__main__":
    make_data(2007)
