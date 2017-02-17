# encoding:utf-8
import numpy as np
import util.bbox as bbox


class Sampler:
    class BatchSampler:
        def __init__(self, sampler):
            if sampler["sampler"]:
                for key in sampler["sampler"].keys():
                    setattr(self, key, sampler["sampler"][key])
            if sampler["sample_constraint"]:
                for key in sampler["sample_constraint"].keys():
                    setattr(self, key, sampler["sample_constraint"][key])
            self.max_trial = sampler["max_trials"]
            self.max_sample = sampler["max_sample"]

    def __init__(self, batch_sampler):
        self.batch_sampler = []
        for sampler in batch_sampler:
            self.batch_sampler.append(Sampler.BatchSampler(sampler))

    def __call__(self, size, BB):
        new_bboxes = []
        src_bbox = np.array([0, 0, size[1], size[0]])
        for sampler in self.batch_sampler:
            if sampler.max_trial == 1:
                new_bboxes.append([src_bbox.copy(), BB.copy()])
                continue
            found = None
            for i in range(sampler.max_trial):
                if found:
                    break
                trans_bbox = self.sample_bbox(sampler)
                new_bbox = self.locate_bbox(src_bbox.copy(), trans_bbox)
                if self.satisfy_constraint(np.array([new_bbox]), BB, sampler):
                    found = True
            if found:
                new_bboxes.append([new_bbox, self.fit_BB(new_bbox, BB.copy())])
            else:
                new_bboxes.append([src_bbox.copy(), BB.copy()])
        return new_bboxes

    def fit_BB(self, src_bbox, BB):
        BB[:, 0][np.where(BB[:, 0] <= src_bbox[0])] = src_bbox[0]
        BB[:, 1][np.where(BB[:, 1] <= src_bbox[1])] = src_bbox[1]
        BB[:, 2][np.where(BB[:, 2] >= src_bbox[2])] = src_bbox[2]
        BB[:, 3][np.where(BB[:, 3] >= src_bbox[3])] = src_bbox[3]
        return BB

    def sample_bbox(self, sampler):
        scale = sampler.min_scale + (sampler.max_scale - sampler.min_scale) * np.random.random()
        min_ar = max(sampler.min_aspect_ratio, np.math.pow(scale, 2))
        max_ar = min(sampler.max_aspect_ratio, 1/np.math.pow(scale, 2))
        aspect_ratio = min_ar + (max_ar - min_ar) * np.random.random()
        bbox_width = scale * np.sqrt(aspect_ratio)
        bbox_height = scale / np.sqrt(aspect_ratio)
        w_off = (1 - bbox_width) * np.random.random()
        h_off = (1 - bbox_height) * np.random.random()
        # print(w_off, h_off, w_off + bbox_width, h_off + bbox_height)
        return (w_off, h_off, w_off + bbox_width, h_off + bbox_height)

    def locate_bbox(self, src_bbox, bbox):
        loc_bbox = np.array([0, 0, 0, 0])
        src_width = src_bbox[2] - src_bbox[0]
        src_height = src_bbox[3] - src_bbox[1]
        loc_bbox[0] = src_bbox[0] + bbox[0] * src_width
        loc_bbox[1] = src_bbox[1] + bbox[1] * src_height
        loc_bbox[2] = src_bbox[0] + bbox[2] * src_width
        loc_bbox[3] = src_bbox[1] + bbox[3] * src_height
        assert loc_bbox[0] < loc_bbox[2]
        assert loc_bbox[1] < loc_bbox[3]
        return loc_bbox

    def satisfy_constraint(self, sample_bbox, object_bboxes, sampler):
        jaccords = bbox.bbox_overlaps2(sample_bbox.astype(np.float), object_bboxes.astype(np.float))
        if hasattr(sampler, "min_jaccard_overlap") and (jaccords < sampler.min_jaccard_overlap).any():
            return False
        if hasattr(sampler, "max_jaccard_overlap") and (jaccords > sampler.max_jaccard_overlap).any():
            return False
        return True
