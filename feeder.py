import pickle
import multiprocessing

import numpy as np
from PIL import Image
from scipy.misc import imresize
import matplotlib.pyplot as plt

import bbox
import batch_sampler

data = pickle.load(open("test_voc2007.pkl", "rb"))
#cropwidth = 256 - model.insize
#voc = [path, (h, w), BBs]

def read_image(model, path, size, BB, label, flip=False):
    image = np.asarray(Image.open(path).convert("RGB"))
    left, top, right, bottom = size
    image = image[top:bottom, left:right, :].astype(np.float32)
    h, w = image.shape[:2]
    BB[:, (0, 2)] -= left
    BB[:, (1, 3)] -= top
    BB[:, (0, 2)] *= 300 / w
    BB[:, (1, 3)] *= 300 / h
    if flip and np.random.randint(2) == 3:
        image[:, :, ::-1]
        BB[:, 0], BB[:, 2] = w - BB[:, 2], right - BB[:, 0]
    image = imresize(image, (300, 300), interp='bicubic')
    loc_mask, conf_mask, loc_t, conf_t = model.make_sample(BB, label)
    image = image.transpose(2, 0, 1).astype(np.float32)
    return image, loc_mask, conf_mask, loc_t, conf_t

class Reader:
    def __init__(self, prior, mean, batch_sampler,  args, insize=300):
        self.prior = prior
        self.insize = insize
        self.prior_num = self.prior.shape[0]
        self.mean = mean
        self.args = args
        self.sampler = Sampler(batch_sampler)

    def __call__(self, path, size, BB, label, flip=False):
        image = np.asarray(Image.open(path).convert("RGB"))
        left, top, right, bottom = size
        image = image[top:bottom, left:right, :].astype(np.float32)
        h, w = image.shape[:2]
        image -= self.mean
        BB[:, (0, 2)] -= left
        BB[:, (1, 3)] -= top
        BB[:, (0, 2)] *= self.insize / w
        BB[:, (1, 3)] *= self.insize / h
        if flip and np.random.randint(2) == 3:
            image[:, :, ::-1]
            BB[:, 0], BB[:, 2] = w - BB[:, 2], right - BB[:, 0]
        image = imresize(image, (self.insize, self.insize), interp='bicubic')
        loc_mask, conf_mask, loc_t, conf_t = self.make_sample(BB, label)
        image = image.transpose(2, 0, 1).astype(np.float32)
        return image, loc_mask, conf_mask, loc_t, conf_t

    def make_sample(self, BB, label):
        loc_mask = np.zeros([self.prior_num, 4])
        conf_mask = np.zeros([self.prior_num, 21])
        overlap = bbox.bbox_overlaps2(self.prior[:, 0] * self.insize, BB)
        positions = np.array(np.where(overlap > 0.5))
        conf_mask[positions[0]] = 1
        loc_mask[positions[0]] = 1

        conf_t = overlap.argmax(1)
        loc_t = np.zeros([self.prior_num, 4])
        loc_t = np.array([self.encoder(BB[conf_t[i]] / self.insize, self.prior[i])
                          if loc_mask[i][0] else [0, 0, 0, 0]
                          for i in range(self.prior_num)])
        conf_t = [label[conf_t[i]] for i in range(self.prior_num)]
        conf_t *= conf_mask[:, 0]
        conf_t += conf_mask[:, 0]
        return [loc_mask, conf_mask, loc_t, conf_t]

    def encoder(self, bbox, prior):
        prior_bbox = prior[0]
        prior_variance = prior[1]
        encode_bbox = np.array([0] * 4, dtype=np.float32)
        prior_width = prior_bbox[2] - prior_bbox[0]
        prior_height = prior_bbox[3] - prior_bbox[1]
        prior_center_x = (prior_bbox[0] + prior_bbox[2]) / 2.
        prior_center_y = (prior_bbox[1] + prior_bbox[3]) / 2.
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        if bbox_width <= 0 or bbox_height <= 0:
            return encode_bbox
        bbox_center_x = (bbox[0] + bbox[2]) / 2.
        bbox_center_y = (bbox[1] + bbox[3]) / 2.
        encode_bbox[0] = (bbox_center_x - prior_center_x) / prior_width / prior_variance[0]
        encode_bbox[1] = (bbox_center_y - prior_center_y) / prior_height / prior_variance[1]
        encode_bbox[2] = np.log(bbox_width / prior_width) / prior_variance[2]
        encode_bbox[3] = np.log(bbox_height / prior_height) / prior_variance[3]
        return encode_bbox


class Feeder:
    def __init__(self, prior, train_list, val_list, mean, batch_sampler, data_q, args, denominator=100000, insize=300):
        self.prior = prior
        self.train_list = train_list
        self.val_list = val_list
        self.insize = insize
        self.prior_num = self.prior.shape[0]
        self.mean = mean
        self.args = args
        self.data_q = data_q
        self.denominator = denominator
        self.sampler = Sampler(batch_sampler)
        self.reader = Reader(prior, mean, batch_sampler, args)

    def __call__(self):
    # Data feeder
        args = self.args
        i = 0
        count = 0
        img_batch = np.ndarray(
            (args.batchsize, 3, self.insize, self.insize), dtype=np.float32)
        conf_batch = np.ndarray(
            (args.batchsize, self.prior_num), dtype=np.int32)
        loc_batch = np.ndarray(
            (args.batchsize, self.prior_num, 4), dtype=np.float32)
        conf_mask = np.ndarray(
            (args.batchsize, self.prior_num, 21), dtype=np.float32)
        loc_mask = np.ndarray(
            (args.batchsize, self.prior_num, 4), dtype=np.float32)

        val_img_batch = np.ndarray(
            (args.val_batchsize, 3, self.insize, self.insize), dtype=np.float32)
        val_conf_batch = np.ndarray(
            (args.val_batchsize, self.prior_num), dtype=np.int32)
        val_loc_batch = np.ndarray(
            (args.val_batchsize, self.prior_num, 4), dtype=np.float32)
        val_conf_mask = np.ndarray(
            (args.val_batchsize, self.prior_num, 21), dtype=np.float32)
        val_loc_mask = np.ndarray(
            (args.val_batchsize, self.prior_num, 4), dtype=np.float32)


        batch_pool = [None] * args.batchsize
        val_batch_pool = [None] * args.val_batchsize
        pool = multiprocessing.Pool(args.loaderjob)
        self.data_q.put('train')

        for epoch in range(1, 1 + args.epoch):
            perm = np.random.permutation(len(self.train_list))
            for idx in perm:
                path, size, BB = self.train_list[idx]
                label = BB[:, 1]
                BB = BB[:, 1:]
                sample = self.sampler(size, BB)
                for sample_size, sample_bbox in sample:
                    batch_pool[i] = pool.apply_async(self.reader, (path, sample_size, sample_bbox, label, True))
                    i += 1
                    if i == args.batchsize:
                        for j, x in enumerate(batch_pool):
                            #x_batch[j], y_batch[j] = x.get()
                            img_batch[j], loc_mask[j], conf_mask[j], loc_batch[j], conf_batch[j] = x.get()
                        self.data_q.put((img_batch.copy(), loc_mask.copy(),
                                         conf_mask.copy(), loc_batch.copy(), conf_batch.copy()))
                        i = 0
                    count += 1
                    if count % self.denominator == 0:
                        self.data_q.put('val')
                        j = 0
                        for path, size, BB in self.val_list:
                            size = [0, 0, size[1], size[0]]
                            label = BB[:, 1]
                            BB = BB[:, 1:]
                            val_batch_pool[j] = pool.apply_async(self.read_image, (path, size, BB, label, False))
                            j += 1
                            if j == args.val_batchsize:
                                for k, x in enumerate(val_batch_pool):
                                    #val_x_batch[k], val_y_batch[k] = x.get()
                                    pass
                                #self.data_q.put((val_x_batch.copy(), val_y_batch.copy()))
                                j = 0

                        self.data_q.put('train')
        pool.close()
        pool.join()
        self.data_q.put('end')

class Sampler:
    class BatchSampler:
        def __init__(self, sampler):
            if sampler["sampler"]:
                for key in sampler["sampler"].keys():
                    setattr(self, key, sampler["sampler"][key])
            if sampler["sample_constraint"]:
                for key in sampler["sampler"].keys():
                    setattr(self, key, sampler["sampler"][key])
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
                new_bboxes.append([new_bbox, self.fit_BB(new_bbox, BB.copy())] )
            else:
                new_bboxes.append([src_bbox.copy(), BB.copy()])
        return new_bboxes

    def fit_BB(self, src_bbox, BB):
        BB[:, 0][np.where(BB[:,0] < src_bbox[0])] = src_bbox[0]
        BB[:, 1][np.where(BB[:,1] < src_bbox[1])] = src_bbox[1]
        BB[:, 2][np.where(BB[:,2] > src_bbox[2])] = src_bbox[2]
        BB[:, 3][np.where(BB[:,3] > src_bbox[3])] = src_bbox[3]
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
        return (w_off, h_off, w_off + bbox_width, h_off + bbox_height)

    def locate_bbox(self, src_bbox, bbox):
        loc_bbox = np.array([0]*4)
        src_width = src_bbox[2] - src_bbox[0]
        src_height = src_bbox[3] - src_bbox[1]
        loc_bbox[0] = src_bbox[0] + bbox[0] * src_width
        loc_bbox[1] = src_bbox[1] + bbox[1] * src_height
        loc_bbox[2] = src_bbox[0] + bbox[2] * src_width
        loc_bbox[3] = src_bbox[1] + bbox[3] * src_height;
        return loc_bbox

    def satisfy_constraint(self, sample_bbox, object_bboxes, sampler):
        jaccords = bbox.bbox_overlaps2(sample_bbox.astype(np.float), object_bboxes.astype(np.float))
        if hasattr(sampler, "min_jaccard_overlap") and (jaccords < sampler.min_jaccard_overlap).sum():
            return False
        if hasattr(sampler, "max_jaccard_overlap") and (jaccords > sampler.max_jaccard_overlap).sum():
            return False
        return True

def test(image, BB):
    image = image.transpose(1,2,0)
    plt.imshow(image)
    currentAxis = plt.gca()
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    for i in BB:
        x1, y1, x2, y2 = i
        x1 = int(round(x1 * image.shape[1]))
        x2 = int(round(x2 * image.shape[1]))
        y1 = int(round(y1 * image.shape[0]))
        y2 = int(round(y2 * image.shape[0]))
        label_name = "a"
        display_txt = '%s: %.2f' % (label_name, 0.1)
        coords = (x1, y1), x2-x1+1, y2-y1+1
        color = colors[1]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(x1, y1, display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    plt.show()

