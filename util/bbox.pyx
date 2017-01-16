#! -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from libcpp cimport bool


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
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )
        for h in range(H):
            for w in range(W):
                for d in range(D_num):
                    iw = (
                        min(boxes[h, w, d, 2], query_boxes[k, 2]) -
                        max(boxes[h, w, d, 0], query_boxes[k, 0])
                    )
                    if iw > 0:
                        ih = (
                            min(boxes[h, w, d, 3], query_boxes[k, 3]) -
                            max(boxes[h, w, d, 1], query_boxes[k, 1])
                        )
                        if ih > 0:
                            ua = float(
                                (boxes[h, w, d, 2] - boxes[h, w, d, 0]) *
                                (boxes[h, w, d, 3] - boxes[h, w, d, 1]) +
                                box_area - iw * ih
                            )
                            overlaps[h, w, d, k] = iw * ih / ua
    return overlaps

def bbox_overlaps2(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):

    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, h, w, d
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )

        if box_area <= 0:
            continue
        for n in range(N):
                    iw = (
                        min(boxes[n, 2], query_boxes[k, 2]) -
                        max(boxes[n, 0], query_boxes[k, 0])
                    )
                    if iw > 0:
                        ih = (
                            min(boxes[n, 3], query_boxes[k, 3]) -
                            max(boxes[n, 1], query_boxes[k, 1])
                        )
                        if ih > 0:
                            ua = float(
                                (boxes[n, 2] - boxes[n, 0]) *
                                (boxes[n, 3] - boxes[n, 1]) +
                                box_area - iw * ih
                            )
                            overlaps[n, k] = iw * ih / ua
    return overlaps

def bbox_overlaps3(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=1] query_boxes):

    cdef unsigned int N = boxes.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1] overlaps = np.zeros((N), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, h, w, d

    box_area = (
        (query_boxes[2] - query_boxes[0] + 1) *
        (query_boxes[3] - query_boxes[1] + 1)
    )
    for n in range(N):
        iw = (
            min(boxes[n, 2], query_boxes[2]) -
            max(boxes[n, 0], query_boxes[0]) + 1
        )
        if iw > 0:
            ih = (
                min(boxes[n, 3], query_boxes[3]) -
                max(boxes[n, 1], query_boxes[1]) + 1
            )
            if ih > 0:
                ua = float(
                    (boxes[n, 2] - boxes[n, 0] + 1) *
                    (boxes[n, 3] - boxes[n, 1] + 1) +
                    box_area - iw * ih
                )
                overlaps[n] = iw * ih / ua
    return overlaps

def nms(
        np.ndarray[DTYPE_t, ndim=2] bboxes,
        np.ndarray[DTYPE_t, ndim=1] scores,
        float nms_th,
        int top_k=300):
    cdef unsigned int score_iter, idx
    cdef np.ndarray[DTYPE_t, ndim=1] IoUs
    cdef np.ndarray[DTYPE_t, ndim=1] cand_bbox
    score_index = scores.argsort()[::-1]
    indices = []
    for idx in score_index:
        if len(indices) >= top_k:
            break
        cand_bbox = bboxes[idx]
        if cand_bbox[0] == cand_bbox[2] or cand_bbox[1] == cand_bbox[3]:
            continue
        IoUs = bbox_overlaps3(bboxes[indices], cand_bbox)
        if np.any(IoUs > nms_th):
            continue
        indices.append(idx)
    return indices
