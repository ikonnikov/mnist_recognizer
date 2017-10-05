# -*- coding: utf-8 -*-

import os
import struct
import numpy as np


def read(dataset='training', path='.'):
    if dataset is 'training':
        img_fname = os.path.join(path, 'train-images.idx3-ubyte')
        lbl_fname = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is 'testing':
        img_fname = os.path.join(path, 't10k-images.idx3-ubyte')
        lbl_fname = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("unknown dataset")

    with open(lbl_fname, 'rb') as lbl_file:
        magic, num = struct.unpack('>II', lbl_file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))

        labels = np.fromfile(lbl_file, dtype=np.int8)

    with open(img_fname, 'rb') as img_file:
        magic, num, rows, cols = struct.unpack('>IIII', img_file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))

        images = np.fromfile(img_file, dtype=np.uint8).reshape(len(labels), rows, cols)

    return (images, labels)
