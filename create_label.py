import numpy as np
import bbox

def create_train_list(priors, labels):
    train_list = []

    for data in labels:
        path, BBs = data
        label = BBs[:,0]
        train_sample = []
        for pri in priors:
            prior = pri[:, :, :,0]
            ans = np.zeros(prior.shape)
            conf_mask = np.zeros(prior.shape)
            overlap = bbox.bbox_overlaps(prior, data)
            positions = np.array(np.where(overlap > 0.5)).transpose(1,0)
            for pos in positions:
                train_sample.append()


