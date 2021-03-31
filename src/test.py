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
import pycocotools._mask as _mask
import cvtools
from collections import defaultdict
import copy

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from torch.utils.data.dataloader import default_collate
import poly_nms


def test_collate_fn(batch):
    batch[0][1] = default_collate([batch[0][1]])
    if len(batch[0]) > 2 and len(batch[0][1]) > 2:
        batch[0][1]['meta']['gt'] = batch[0][2]
        del batch[0][2:]
    return batch[0]


def read_dota_image(img_info, mode='train'):
    # if mode == 'test':
    #     file_name = img_info['filename']
    # else:
    #     file_name = img_info['file_name']
    file_name = img_info['file_name']
    filename = osp.splitext(file_name)[0]
    suffix = osp.splitext(file_name)[1]
    crop_str = list(map(str, img_info['crop']))
    crop_img_path = osp.join('/code/data/DOTA/crop1024',
                             '_'.join([filename] + crop_str) + suffix)
    if not osp.isfile(crop_img_path):
        img_path = os.path.join('/code/data/DOTA/{}/images'.format(mode), file_name)
        image = cv2.imread(img_path)
        sx, sy, ex, ey = img_info['crop']
        image = image[sy:ey+1, sx:ex+1]
        cv2.imwrite(crop_img_path, image)
    else:
        image = cv2.imread(crop_img_path)
    return image


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
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
        image = read_dota_image(img_info, mode=self.dataset.split)

        anns = None
        if hasattr(self.dataset, 'coco'):
            ann_ids = self.dataset.coco.getAnnIds(imgIds=[img_id])
            # 如果不deep拷贝，修改了anns就修改了self.coco里的标签
            anns = copy.deepcopy(self.dataset.coco.loadAnns(ids=ann_ids))

        images, meta = {}, {}
        for scale in opt.test_scales:
            if opt.task == 'ddd':
                images[scale], meta[scale] = self.pre_process_func(
                    image, scale, img_info['calib'])
            else:
                images[scale], meta[scale] = self.pre_process_func(
                    image, scale, anns=anns)
        return [img_id, {'images': images, 'image': image, 'meta': meta}, anns]

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
        img_id = self.images[idx]
        img_info = self.dataset.coco.loadImgs(ids=[img_id])[0]
        # img_info = self.images[idx]
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
    for cls_id, det_cls in ret['results'].items():
        det_cls = det_cls[det_cls[:, -1] > 0.1]
        if len(det_cls) == 0: continue
        ids = poly_nms.poly_gpu_nms(det_cls, 0.15)
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


def prefetch_test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    loger = Logger(opt)
    Detector = detector_factory[opt.task]

    split = 'val' if not opt.test else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
        collate_fn=test_collate_fn
    )

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        # pre_processed_images['img_info'] = dataset.coco.loadImgs([img_id])[0]
        ret = detector.run(pre_processed_images)
        results[img_id] = ret['results']
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        # if opt.debug:
        #     # img = pre_processed_images['image'].detach().cpu().numpy().squeeze()
        #     print(type(ret['results']))
        #     vis_dets(ret, img_id, dataset, split=split)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix += '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
                t, tm=avg_time_stats[t])
        bar.next()
    bar.finish()
    sava_path = opt.save_dir + '/{}_results'.format(split)
    dataset.save_pkl(results, sava_path)
    dataset.save_dota_format(results, sava_path)
    if split != 'test':
        dataset.dota_eval(sava_path + '/Task1_{:s}.txt')
        mean_ap, eval_results = dataset.coco_eval(results)
        print(mean_ap, eval_results)


def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    Detector = detector_factory[opt.task]

    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])

        if opt.task == 'ddd':
            ret = detector.run(img_path, img_info['calib'])
        else:
            ret = detector.run(img_path)

        results[img_id] = ret['results']

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()
    dataset.run_eval(results, opt.save_dir)


if __name__ == '__main__':
    # os.system("/opt/conda/bin/python /root/DOTA_devkit/dota_evaluation_task1.py")
    torch.backends.cudnn.benchmark = True
    opt = opts().parse()
    if opt.val_result:
        Dataset = dataset_factory[opt.dataset]
        opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
        print(opt)
        split = 'val' if not opt.trainval else 'test'
        dataset = Dataset(opt, split)
        # results = cvtools.load_pkl(opt.save_dir + '/{}_results'.format(split) + '/dets.pkl')
        results = cvtools.load_pkl('exp/ctdet/dota_dla/val_results/dets.pkl')
        dataset.save_dota_format(results, opt.save_dir + '/{}_results'.format(split))
        dataset.dota_eval(opt.save_dir + '/{}_results'.format(split) + '/Task1_{:s}.txt')
        # dataset.coco_eval(results)
    else:
        if opt.not_prefetch_test:
            test(opt)
        else:
            prefetch_test(opt)
