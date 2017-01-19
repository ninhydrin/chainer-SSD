#! -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def decoder(
        np.ndarray[DTYPE_t, ndim=2] loc,
        np.ndarray[DTYPE_t, ndim=3] prior):

    cdef np.ndarray[DTYPE_t, ndim=2] bbox_data = np.zeros((loc.shape[0], 4),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] prior_center_x = (prior[:, 0, 2] + prior[:, 0, 0])/2
    cdef np.ndarray[DTYPE_t, ndim=1] prior_center_y = (prior[ :, 0, 3] + prior[:, 0, 1])/2
    cdef np.ndarray[DTYPE_t, ndim=1] prior_width = prior[:, 0, 2] - prior[:, 0, 0]
    cdef np.ndarray[DTYPE_t, ndim=1] prior_height = prior[:, 0, 3] - prior[:, 0, 1]

    cdef np.ndarray[DTYPE_t, ndim=1] decode_bbox_center_x = prior[:, 1, 0] * loc[:, 0] * prior_width + prior_center_x
    cdef np.ndarray[DTYPE_t, ndim=1] decode_bbox_center_y = prior[:, 1, 1] * loc[:, 1] * prior_height + prior_center_y
    cdef np.ndarray[DTYPE_t, ndim=1] decode_bbox_width = np.exp(prior[:, 1, 2] * loc[:, 2]) * prior_width
    cdef np.ndarray[DTYPE_t, ndim=1] decode_bbox_height = np.exp(prior[:, 1, 3] * loc[:, 3]) * prior_height
    bbox_data[:, 0] = decode_bbox_center_x - decode_bbox_width / 2.
    bbox_data[:, 1] = decode_bbox_center_y - decode_bbox_height / 2.
    bbox_data[:, 2] = decode_bbox_center_x + decode_bbox_width / 2.
    bbox_data[:, 3] = decode_bbox_center_y + decode_bbox_height / 2.
    return bbox_data

def encoder(
        np.ndarray[DTYPE_t, ndim=2] loc,
        np.ndarray[DTYPE_t, ndim=3] prior):
    cdef np.ndarray[DTYPE_t, ndim=2] encode_bbox = np.zeros((loc.shape[0], 4))
    cdef np.ndarray[DTYPE_t, ndim=1] prior_center_x = (prior[:, 0, 2] + prior[:, 0, 0])/2
    cdef np.ndarray[DTYPE_t, ndim=1] prior_center_y = (prior[ :, 0, 3] + prior[:, 0, 1])/2
    cdef np.ndarray[DTYPE_t, ndim=1] prior_width = prior[:, 0, 2] - prior[:, 0, 0]
    cdef np.ndarray[DTYPE_t, ndim=1] prior_height = prior[:, 0, 3] - prior[:, 0, 1]
    cdef np.ndarray[DTYPE_t, ndim=1] bbox_width = loc[:, 2] - loc[:, 0] 
    cdef np.ndarray[DTYPE_t, ndim=1] bbox_height = loc[:, 3] - loc[:, 1]

    bbox_width[np.where(bbox_width <= 0)] = 1
    bbox_height[np.where(bbox_height <= 0)] = 1

    cdef np.ndarray[DTYPE_t, ndim=1] bbox_center_x = (loc[:, 0] + loc[:, 2] ) / 2.
    cdef np.ndarray[DTYPE_t, ndim=1] bbox_center_y = (loc[:, 1] + loc[:, 3] ) / 2. 

    encode_bbox[:, 0] = (bbox_center_x - prior_center_x) / prior_width / prior[:, 1, 0]
    encode_bbox[:, 1] = (bbox_center_y - prior_center_y) / prior_height / prior[:, 1, 1]
    encode_bbox[:, 2] = np.log(bbox_width / prior_width) / prior[:, 1, 2]
    encode_bbox[:, 3] = np.log(bbox_height / prior_height) / prior[:, 1, 3]
    return encode_bbox
