import matplotlib.pyplot as plt
import numpy as np

import bbox
import ssd

labelmap = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def detect(prior, loc, conf, nms_th=0.45, cls_th=0.6):
    cand = []
    loc = ssd.decoder(loc, prior)
    for label in range(1, 21):
        cand_score = np.where(conf[:, label] > cls_th)
        scores = conf[:, label][cand_score]
        cand_loc = loc[cand_score]
        k = bbox.nms(cand_loc, scores, nms_th, 300)
        for i in k:
            cand.append(np.hstack([[label], [scores[i]], cand_loc[i]]))
    cand = np.array(cand)
    return cand


def draw(image, cand, f_name=None):
    plt.imshow(image)
    currentAxis = plt.gca()
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    for i in cand:
        label, conf, x1, y1, x2, y2 = i
        label = int(label) - 1
        x1 = int(round(x1 * image.shape[1]))
        x2 = int(round(x2 * image.shape[1]))
        y1 = int(round(y1 * image.shape[0]))
        y2 = int(round(y2 * image.shape[0]))
        label_name = labelmap[int(label)]
        display_txt = '%s: %.2f' % (label_name, conf)
        coords = (x1, y1), x2-x1+1, y2-y1+1
        color = colors[int(label)]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(x1, y1, display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    if f_name:
        plt.show()
    else:
        plt.savefig(f_name)
