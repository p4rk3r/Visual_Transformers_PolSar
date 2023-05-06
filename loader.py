# This is a PyTorch implementation of ViT for PolSAR image classification
# This repo aims to be minimal modifications on the official PyTorch ImageNet training code
# see https://github.com/pytorch/examples
# Copyright
# p4rk3r@p4rk3r.io
# P4rk3r industries


import numpy as np
import scipy.io as sio
from torch.utils.data.dataset import Dataset


class VITDataset(Dataset):
    def __init__(self, train_lines):
        super(VITDataset, self).__init__()
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.flag = True

    def __len__(self):
        return self.train_batches

    def get_random_data(self, annotation_line):
        line = annotation_line.split()
        # Read .mat files
        image = sio.loadmat(line[0])
        image = image['need_data']
        target = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        return image, target

    def __getitem__(self, index):
        lines = self.train_lines
        img, y = self.get_random_data(lines[index])
        img = np.array(img, dtype=np.float32)
        tmp_inp = np.transpose(img, (2, 0, 1))
        y = np.squeeze(y)
        tmp_targets = np.array(y, dtype=np.int32)
        return tmp_inp, tmp_targets
