from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import os.path as osp
import cv2
import numpy as np
from progress.bar import Bar
import torch
import torch.utils.data
import pycocotools._mask as _mask
import cvtools
from collections import defaultdict
import copy
import ctypes

from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
# from detectors.detector_factory import detector_factory
from torch.utils.data.dataloader import default_collate
from external.nms import soft_nms
from opts import opts
from logger import Logger
import poly_nms

lib = ctypes.cdll.LoadLibrary('/code/RCNet_TensorRT/lib/libdetector.so')


def test_collate_fn(batch):
    batch[0][1] = default_collate([batch[0][1]])
    if len(batch[0]) > 2 and len(batch[0][1]) > 2:
        batch[0][1]['meta']['gt'] = batch[0][2]
        del batch[0][2:]
    return batch[0]


def get_image_path(img_info, mode='train'):
    if mode == 'test':
        file_name = img_info['filename']
    else:
        file_name = img_info['file_name']

    filename = osp.splitext(file_name)[0]
    suffix = osp.splitext(file_name)[1]
    crop_str = list(map(str, img_info['crop']))
    crop_img_path = osp.join('/media/data/DOTA/crop',
                             '_'.join([filename] + crop_str) + suffix)
    if not osp.isfile(crop_img_path):
        return os.path.join('/media/data/DOTA/{}/images'.format(mode), file_name)
    else:
        return crop_img_path


def read_dota_image(img_info, mode='train'):
    if mode == 'test':
        file_name = img_info['filename']
    else:
        file_name = img_info['file_name']

    filename = osp.splitext(file_name)[0]
    suffix = osp.splitext(file_name)[1]
    crop_str = list(map(str, img_info['crop']))
    crop_img_path = osp.join('/media/data/DOTA/crop',
                             '_'.join([filename] + crop_str) + suffix)
    if not osp.isfile(crop_img_path):
        img_path = os.path.join('/media/data/DOTA/{}/images'.format(mode), file_name)
        image = cv2.imread(img_path)
        sx, sy, ex, ey = img_info['crop']
        image = image[sy:ey+1, sx:ex+1]
        cv2.imwrite(crop_img_path, image)
    else:
        image = cv2.imread(crop_img_path)
    return image


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func=None):
        self.dataset = dataset
        self.pre_process_func = pre_process_func
        self.opt = opt
        self.test_mode = dataset.split == 'test'
        self.images = dataset.images
        self.img_dir = dataset.img_dir
        if not self.test_mode:
            self.load_image_func = dataset.coco.loadImgs

    def _get_dota_item(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        image = get_image_path(img_info, mode=self.dataset.split)
        return [img_id, image]

    def prepare_train_img(self, index):
        if self.opt.dataset == 'dota':
            return self._get_dota_item(index)
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        images, meta = {}, {}
        for scale in opt.test_scales:
            if opt.task == 'ddd':
                images[scale], meta[scale] = self.pre_process_func(
                    image, scale, img_info['calib'])
            else:
                images[scale], meta[scale] = self.pre_process_func(image, scale)
        return img_id, {'images': images, 'image': image, 'meta': meta}

    def prepare_test_img(self, idx):
        img_info = self.images[idx]
        image = read_dota_image(img_info, mode=self.dataset.split)
        images, meta = {}, {}
        for scale in opt.test_scales:
            images[scale], meta[scale] = self.pre_process_func(image, scale)
        return [idx, {'images': images, 'image': image, 'meta': meta}]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = np.random.choice(len(self.dataset) - 1)
                continue
            return data

    def __len__(self):
        return len(self.images)


def vis_dets(ret, img_id, dataset, save_by_cat=False, split='val'):
    if split != 'test':
        img_info = dataset.coco.loadImgs([img_id])[0]
        ann_ids = dataset.coco.getAnnIds(imgIds=[img_id])
        anns = dataset.coco.loadAnns(ids=ann_ids)
        file_name = img_info['file_name']
    else:
        img_info = dataset.images(img_id)
        file_name = img_info['filename']
    colors = []
    img = read_dota_image(img_info, split)
    for cls_id, det_cls in ret.items():
        if len(det_cls) == 0: continue
        det_cls = det_cls[det_cls[:, -1] > 0.1]
        if len(det_cls) == 0: continue
        ids = poly_nms.poly_gpu_nms(det_cls.astype(np.float32), 0.15)
        det_cls = det_cls[ids]
        text = [dataset.class_name[cls_id] + str(round(score, 2))
                for score in det_cls[..., -1]]
        img = cvtools.draw_boxes_texts(img, det_cls[:, :-1],
                                       texts=None,
                                       line_width=2,
                                       colors=[dataset.voc_color[cls_id-1]] * len(det_cls),
                                       box_format="polygon")
    crop_str = list(map(str, img_info['crop']))
    filename = osp.splitext(file_name)[0]
    suffix = osp.splitext(file_name)[1]
    save_img_name = osp.join('_'.join([filename] + crop_str) + suffix)
    if save_by_cat and split != 'test':
        cats = {ann['category_id'] for ann in anns}
        for cat in cats:
            cat_name = dataset.coco.cats[cat]['name']
            file = osp.join(opt.debug_dir, cat_name, save_img_name)
            cvtools.imwrite(img, file)
    else:
        file = osp.join(opt.debug_dir, save_img_name)
        cvtools.imwrite(img, file)


def rotate_tect(x1, y1, x3, y3, angle):
    x2 = x3
    y2 = y1
    x4 = x1
    y4 = y3
    x = (x1 + x3) / 2.
    y = (y1 + y3) / 2.

    angle = 1. / (1. + np.exp(-angle))
    angle = 2. * angle - 1.
    cosA = np.cos(np.pi / 2. * angle)
    sinA = np.sin(np.pi / 2. * angle)

    x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
    y1n = (x1 - x) * sinA + (y1 - y) * cosA + y

    x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
    y2n = (x2 - x) * sinA + (y2 - y) * cosA + y

    x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
    y3n = (x3 - x) * sinA + (y3 - y) * cosA + y

    x4n = (x4 - x) * cosA - (y4 - y) * sinA + x
    y4n = (x4 - x) * sinA + (y4 - y) * cosA + y

    return x1n, y1n, x2n, y2n, x3n, y3n, x4n, y4n


def process_trt_ret(results):
    out = [[] for _ in range(15)]
    num_det = int(results[0])
    for i in range(num_det):
        x1 = results[i*7+1]
        y1 = results[i*7+2]
        x3 = results[i*7+3]
        y3 = results[i*7+4]
        a = results[i*7+5]
        prob = results[i*7+6]
        cls_id = int(results[i*7+7])
        obb = rotate_tect(x1, y1, x3, y3, a)
        out[cls_id].append(list(obb) + [prob])
    run_time = results[num_det*7+1]
    return {i+1: np.array(a).reshape((-1, 9)) for i, a in enumerate(out)}, run_time


def prefetch_test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    loger = Logger(opt)
    # Detector = detector_factory[opt.task]

    split = 'val' if not opt.test else 'test'
    dataset = Dataset(opt, split)
    # detector = Detector(opt)

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
        collate_fn=test_collate_fn
    )

    lib.detector_new.restype = ctypes.POINTER(ctypes.c_void_p)  # 声明函数返回值类型
    lib.detect.restype = ctypes.POINTER(ctypes.c_float)  # 声明函数返回值类型为float*

    model = lib.detector_new(b'/code/RCNet_TensorRT/model/rcnet_dla_int8.engine')
    # model = lib.detector_new(b'/code/RCNet_TensorRT/model/rcnet_dla_fp16.engine')
    # model = lib.detector_new(b'/code/RCNet_TensorRT/model/ctdet_dota_dla_r_fp32.engine')
    # results = lib.detect(model, b'/media/data/DOTA/crop/P0838_0_720_799_1519.png')

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['net']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        # pre_processed_images['img_info'] = dataset.coco.loadImgs([img_id])[0]
        # ret = model.run(pre_processed_images)
        ret = lib.detect(model, bytes(pre_processed_images[0], encoding = "utf8"))
        ret, run_time = process_trt_ret(ret)
        results[img_id] = ret
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        if opt.debug:
            vis_dets(ret, img_id, dataset, split=split)
        avg_time_stats['net'].update(run_time)
        Bar.suffix += '|net {tm.val:.3f}ms ({tm.avg:.3f}ms) '.format(tm=avg_time_stats['net'])
        bar.next()
    bar.finish()
    sava_path = opt.save_dir + '/{}_results'.format(split)
    dataset.save_pkl(results, sava_path)
    dataset.save_dota_format(results, sava_path)
    if split != 'test':
        dataset.dota_eval(sava_path + '/Task1_{:s}.txt')
        mean_ap, eval_results = dataset.coco_eval(results)
        print(mean_ap, eval_results)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    opt = opts().parse()
    prefetch_test(opt)
