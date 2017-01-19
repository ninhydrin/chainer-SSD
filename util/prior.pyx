#! -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t


def prior(size, min_size, max_size, aspect, flip, clip, variance):
    cdef unsigned int height = size[0]
    cdef unsigned int width = size[1]
    cdef float img_height = 300
    cdef float img_width = 300
    cdef unsigned int i = 0
    cdef float step_x = img_width / width
    cdef float step_y = img_width / height
    cdef float j, center_x, center_y, box_width, box_height
    cdef unsigned int h, w, idx
    aspect_ratio = []
    for j in aspect:
        aspect_ratio.append(j)
        aspect_ratio.append(1/j)

    wid_hei_list = [(min_size, min_size)]
    if max_size > 0:
        #second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
        wid_hei_list.append((np.sqrt(min_size * max_size), np.sqrt(min_size * max_size)))
    for ar in aspect_ratio:
        if abs(ar - 1.) < 1e-6:
            continue
        wid_hei_list.append((min_size * np.sqrt(ar), min_size / np.sqrt(ar)))

    cdef np.ndarray[DTYPE_t, ndim=5] top_data = np.zeros([height, width, len(wid_hei_list), 2, 4])

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
