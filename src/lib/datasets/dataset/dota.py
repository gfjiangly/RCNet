# -*- encoding:utf-8 -*-
# @Time    : 2019/10/14 18:12
# @Author  : gfjiang
# @Site    : 
# @File    : dota.py
# @Software: PyCharm
import os
import os.path as osp
import numpy as np
import pycocotools.coco as coco
from tqdm import tqdm
import mmcv
from collections import defaultdict
from multiprocessing import Pool
import poly_nms
import cvtools
# from cvtools.evaluation.merge_dets import MergeCropDetResults
from cvtools.evaluation.eval_crop import EvalCropQuality

from .coco import COCO
from dota_evaluation_task1 import eval_dota_task1


class DOTA(COCO):
    num_classes = 15
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split=None):
        super(COCO, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'DOTA')
        self.img_dir = os.path.join(self.data_dir, 'crop')
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'dota_test_1024.json').format(split)
        # elif split == 'val':
        #     self.annot_path = os.path.join(
        #         self.data_dir, 'annotations',
        #         '{}_dota+crop1024.json').format(split)
        else:
            if opt.task == 'exdet':
                self.annot_path = os.path.join(
                    self.data_dir, 'a nnotations',
                    'instances_extreme_{}2017.json').format(split)
            # else:
            #     self.annot_path = os.path.join(
            #         self.data_dir, 'annotations',
            #         'oversampling/{}_dota+newcrop+over.json').format(split)
            else:
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    '{}_dota_crop800.json').format(split)
            # else:
            #     self.annot_path = os.path.join(
            #         self.data_dir, 'annotations',
            #         'crop800x800/{}_dota+crop800x800.json').format(split)
        self.max_objs = 500
        self.class_name = [
            '__background__', 'large-vehicle', 'swimming-pool', 'helicopter',
            'bridge', 'plane', 'ship', 'soccer-ball-field', 'basketball-court',
            'ground-track-field', 'small-vehicle', 'harbor', 'baseball-diamond',
            'tennis-court', 'roundabout', 'storage-tank']
        self._valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        # if split != 'test':
        #     print('==> initializing coco 2017 {} data.'.format(split))
        #     self.coco = coco.COCO(self.annot_path)
        #     self.images = self.coco.getImgIds()
        # else:
        #     self.images = cvtools.load_json(self.annot_path)
        print('==> initializing dota {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()

        if opt.debug:
            self.images = self.images[:100]
        # self.images = self.images[:100]
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def crop_bbox_map_back(self, bb, crop_start):
        bb_shape = bb.shape
        original_bb = bb.reshape(-1, 2) + np.array(crop_start).reshape(-1, 2)
        return original_bb.reshape(bb_shape)

    def genereteImgResults(self, results):
        """结合子图的结果，映射回原图，应用nms, 生成一张整图的结果"""
        imgResults = defaultdict(list)
        for image_id, dets in results.items():
            img_info = self.coco.imgs[image_id+1]
            labels = mmcv.concat_list([[j]*len(det)for j, det in dets.items()])
            if len(labels) == 0:
                continue
            scores = mmcv.concat_list([det[:, 8] for det in dets.values() if len(det) > 0])
            rbboxes = np.vstack([det[:, :8] for det in dets.values() if len(det) > 0])
            if 'crop' in img_info:
                rbboxes = self.crop_bbox_map_back(rbboxes, img_info['crop'][:2])
            assert len(rbboxes) == len(labels)
            if len(labels) > 0:
                result = [rbboxes, labels, scores]
                imgResults[img_info['file_name']].append(result)
        return imgResults

    def merge_results(self, results, anns=None, n_worker=0):
        # anns = mmcv.load(ann_file)
        # results = mmcv.load(result_file)
        imgResults = defaultdict(list)
        if n_worker > 0:
            pool = Pool(processes=n_worker)
            num = len(anns) // n_worker
            anns_group = [anns[i:i + num] for i in range(0, len(anns), num)]
            results_group = [results[i:i + num] for i in range(0, len(results), num)]
            res = []
            for anns, results in tqdm(zip(anns_group, results_group)):
                res.append(pool.apply_async(self.genereteImgResults, args=(anns, results,)))
            pool.close()
            pool.join()
            for item in res:
                imgResults.update(item.get())
        else:
            imgResults = self.genereteImgResults(results)
        for filename, result in imgResults.items():
            rbboxes = np.vstack([bb[0] for bb in result]).astype(np.int)
            labels = np.hstack([bb[1] for bb in result])
            scores = np.hstack([bb[2] for bb in result])
            dets = np.hstack([rbboxes, scores[:, np.newaxis]]).astype(np.float32)
            ids = self.poly_nms(
                dets, labels,
                # ('soccer-ball-field', 'ground-track-field')
            )
            # rbboxes = np.hstack([rbboxes, labels, scores])
            imgResults[filename] = [rbboxes[ids], labels[ids], scores[ids]]
        return imgResults

    def poly_nms(self, dets, labels, not_nms_cls=None):
        if not_nms_cls is not None:
            not_nms_ids = [
                np.where(labels == self.class_name.index(cls))[0]
                for cls in not_nms_cls
            ]
            nms_ids = [
                np.where(labels != self.class_name.index(cls))[0]
                for cls in not_nms_cls
            ]
            not_nms_ids = np.hstack(not_nms_ids)
            nms_ids = np.hstack(nms_ids)
            nms_dets = dets[nms_ids]
            ids = poly_nms.poly_gpu_nms(nms_dets, 0.15)
            nms_ids = nms_ids[ids]
            new_ids = np.hstack([nms_ids, not_nms_ids])
        else:
            new_ids = poly_nms.poly_gpu_nms(dets, 0.15)
        return new_ids

    def ImgResults2CatResults(self, imgResults):
        catResults = defaultdict(list)
        for filename in imgResults:
            rbboxes = imgResults[filename][0]
            cats = imgResults[filename][1]
            scores = imgResults[filename][2]
            for ind, cat in enumerate(cats):
                catResults[cat].append([filename, scores[ind], rbboxes[ind]])
        return catResults

    def writeResults2DOTATestFormat(self, catResults, class_names, save_path):
        for cat_id, result in catResults.items():
            lines = []
            for filename, score, rbbox in result:
                filename = osp.splitext(filename)[0]
                bbox = list(map(str, list(rbbox)))
                score = str(round(score, 3))
                lines.append(' '.join([filename] + [score] + bbox))
            cvtools.write_list_to_file(
                lines, osp.join(save_path, 'Task1_' + class_names[cat_id] + '.txt'))

    def save_pkl(self, results, save_dir):
        # imgResults = self.merge_results(self.coco.anns, results)
        # catResults = self.ImgResults2CatResults(imgResults)
        # self.writeResults2DOTATestFormat(catResults, self.class_name, save_dir)
        cvtools.dump_pkl(results, save_dir+'/dets.pkl')

    def save_dota_format(self, results, save_dir):
        # dets = MergeCropDetResults(self.coco, results, num_coors=8)
        # dets.merge(poly_nms.poly_gpu_nms)
        # dets.save_dota_det_format(save_dir)
        imgResults = self.merge_results(results)
        catResults = self.ImgResults2CatResults(imgResults)
        self.writeResults2DOTATestFormat(catResults, self.class_name, save_dir)

    def dota_eval(self, result_files):
        eval_dota_task1(result_files, det_thresh=0.01)

    def coco_eval(self, results):
        ann_file = '/media/data/DOTA/annotations/val_dota_original.json'
        eval_crop_quality = EvalCropQuality(ann_file, self.coco, results, num_coors=8)
        mean_ap, eval_results = eval_crop_quality.eval()
