import csv
import os
import os.path
import tarfile
import torch.utils.data as data
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
import pickle
import glob
from collections import defaultdict

fn_map = {}
for fn in glob.glob("images/*.jpg"):
    tmp = fn.split('_')[1]
    fn_map[tmp] = fn


def read_info(root, set):
    imagelist = {}
    hash2ids = {}
    if set == "trainval": 
        path = os.path.join(root, "ImageList", "TrainImagelist.txt")
    elif set == "test":
        path = os.path.join(root, "ImageList", "TestImagelist.txt")
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.split('\\')[-1]
            start = line.index('_')
            end = line.index('.')
            imagelist[i] = line[start+1:end]
            hash2ids[line[start+1:end]] = i

    return imagelist


def read_object_labels_csv(file, imagelist, fn_map, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = int(row[0])
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                name2 = fn_map[imagelist[name]]
                item = (name2, labels)
                images.append(item)
            rownum += 1
    return images


class NUSWIDEClassification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None):
        self.root = root
        self.path_images = os.path.join(root, 'images')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # define path of csv file
        path_csv = os.path.join(self.root, 'classification_labels')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')
        imagelist = read_info(root, set)

        self.classes = 81
        self.images = read_object_labels_csv(file_csv, imagelist, fn_map)

        print('[dataset] NUSWIDE classification set=%s number of classes=%d  number of images=%d' % (
            set, self.classes, len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path, self.inp), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
