#! -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def bbox_overlaps(
        np.ndarray[DTYPE_t, ndim=4] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):

    cdef unsigned int H = boxes.shape[0]
    cdef unsigned int W = boxes.shape[1]
    cdef unsigned int D_num = boxes.shape[2]

    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=4] overlaps = np.zeros((H, W, D_num, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, h, w, d
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for h in range(H):
            for w in range(W):
                for d in range(D_num):
                    iw = (
                        min(boxes[h, w, d, 2], query_boxes[k, 2]) -
                        max(boxes[h, w, d, 0], query_boxes[k, 0]) + 1
                    )
                    if iw > 0:
                        ih = (
                            min(boxes[h, w, d, 3], query_boxes[k, 3]) -
                            max(boxes[h, w, d, 1], query_boxes[k, 1]) + 1
                        )
                        if ih > 0:
                            ua = float(
                                (boxes[h, w, d, 2] - boxes[h, w, d, 0] + 1) *
                                (boxes[h, w, d, 3] - boxes[h, w, d, 1] + 1) +
                                box_area - iw * ih
                            )
                            overlaps[h, w, d, k] = iw * ih / ua
    return overlaps
