# coding: utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from torch.utils.data.dataloader import default_collate

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.dota import DOTA

dataset_factory = {
    'dota': DOTA,
    'coco': COCO,
    'pascal': PascalVOC,
    'kitti': KITTI,
    'coco_hp': COCOHP
}

_sample_factory = {
    'exdet': EXDetDataset,
    'ctdet': CTDetDataset,
    'ddd': DddDataset,
    'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass    # 仅仅为了组合两个类
    return Dataset


def collate_fn(batch):
    max_shape = np.max(np.array([data[0]['input'].shape[1:] for data in batch]), axis=0)
    for i, data in enumerate(batch):
        img = data[0]['input']
        hm = data[0]['hm']
        ind = data[0]['ind']

        pad = np.empty(np.array([3]+max_shape.tolist()), dtype=img.dtype)
        pad[...] = 0
        pad[..., :img.shape[1], :img.shape[2]] = img
        batch[i][0]['input'] = pad

        for j in range(len(ind)):
            y = ind[j] // hm.shape[2]
            x = ind[j] - y * hm.shape[2]
            ind[j] = y * (pad.shape[1] // 4) + x
        batch[i][0]['ind'] = ind

        pad = np.empty(np.array([hm.shape[0]]+(max_shape // 4).tolist()),
                       dtype=hm.dtype)
        pad[...] = 0
        pad[..., :hm.shape[1], :hm.shape[2]] = hm
        batch[i][0]['hm'] = pad
    return default_collate(batch)
