from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch
import cv2.cv2 as cv
from models.utils import _sigmoid

try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode, ctdet_decode_replace, trans_gt
from models.utils import flip_tensor
from utils.post_process import ctdet_post_process
from .base_detector import BaseDetector
from src.lib.datasets.sample.ctdet import convert_angle


def replace_ht(anns, output, in_shape, num_cls=15, a_method=1):
    new_h = output['hm'].shape[2]
    new_w = output['hm'].shape[3]
    down_ratio_h = in_shape[0] / new_h
    down_ratio_w = in_shape[1] / new_w
    hm = torch.zeros((1, num_cls, new_h, new_w), dtype=torch.float32).cuda()
    offset = output['reg']
    for ann in anns:
        cls = int(ann['category_id']) - 1
        segm = np.array(ann['segmentation'][0])
        segm_hull = cv.convexHull(
            segm.reshape(-1, 2).astype(np.float32), clockwise=False
        )
        xy, _, _ = cv.minAreaRect(segm_hull)
        ct = xy[0] / down_ratio_w, xy[1] / down_ratio_h
        ct_int = int(ct[0]), int(ct[1])
        try:
            offset[0][0][ct_int[1]][ct_int[0]] = ct[0] - ct_int[0]
            offset[0][1][ct_int[1]][ct_int[0]] = ct[1] - ct_int[1]
            hm[0][cls][ct_int[1]][ct_int[0]] = 1.
        except IndexError:
            print(ann)
    output['hm'] = hm


def replace_ht_wh(anns, output, in_shape, num_cls=15, a_method=1):
    new_h = output['hm'].shape[2]
    new_w = output['hm'].shape[3]
    down_ratio_h = in_shape[0] / new_h
    down_ratio_w = in_shape[1] / new_w
    hm = torch.zeros((1, num_cls, new_h, new_w), dtype=torch.float32).cuda()
    wh = output['wh']
    offset = output['reg']
    for ann in anns:
        cls = int(ann['category_id']) - 1
        segm = np.array(ann['segmentation'][0])
        segm_hull = cv.convexHull(
            segm.reshape(-1, 2).astype(np.float32), clockwise=False)
        xy, (w, h), a = cv.minAreaRect(segm_hull)
        a, w, h = convert_angle(a, w, h, a_method)
        ct = xy[0] / down_ratio_w, xy[1] / down_ratio_h
        ct_int = int(ct[0]), int(ct[1])
        try:
            offset[0][0][ct_int[1]][ct_int[0]] = ct[0] - ct_int[0]
            offset[0][1][ct_int[1]][ct_int[0]] = ct[1] - ct_int[1]
            hm[0][cls][ct_int[1]][ct_int[0]] = 1.
            wh[0][0][ct_int[1]][ct_int[0]] = w / down_ratio_w
            wh[0][1][ct_int[1]][ct_int[0]] = h / down_ratio_h
        except IndexError:
            print(ann)
    output['hm'] = hm


def replace_ht_a(anns, output, in_shape, num_cls=15, a_method=1):
    new_h = output['hm'].shape[2]
    new_w = output['hm'].shape[3]
    down_ratio_h = in_shape[0] / new_h
    down_ratio_w = in_shape[1] / new_w
    hm = torch.zeros((1, num_cls, new_h, new_w), dtype=torch.float32).cuda()
    angle = output['a']
    offset = output['reg']
    for ann in anns:
        cls = int(ann['category_id']) - 1
        segm = np.array(ann['segmentation'][0])
        segm_hull = cv.convexHull(
            segm.reshape(-1, 2).astype(np.float32), clockwise=False)
        xy, (w, h), a = cv.minAreaRect(segm_hull)
        a, w, h = convert_angle(a, w, h, a_method)
        ct = xy[0] / down_ratio_w, xy[1] / down_ratio_h
        ct_int = int(ct[0]), int(ct[1])
        try:
            offset[0][0][ct_int[1]][ct_int[0]] = ct[0] - ct_int[0]
            offset[0][1][ct_int[1]][ct_int[0]] = ct[1] - ct_int[1]
            hm[0][cls][ct_int[1]][ct_int[0]] = 1.
            angle[0][0][ct_int[1]][ct_int[0]] = a / 90.
        except IndexError:
            print(ann)
    output['hm'] = hm


class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def process(self, images, gt=None, return_time=False):
        with torch.no_grad():
            outputs = self.model(images)[-1]
            dets = []
            for output in outputs:
                if self.opt.dataset == 'dota':
                    output['a'].sigmoid_()
                    if self.opt.a_method == 1:
                        output['a'] = 2. * output['a'] - 1.

                if self.opt.replace_hm and self.opt.replace_wh:
                    replace_ht_wh(gt, output, images.shape[2:])
                elif self.opt.replace_hm and self.opt.replace_a:
                    replace_ht_a(gt, output, images.shape[2:])
                elif self.opt.replace_hm:
                    replace_ht(gt, output, images.shape[2:])
                else:
                    output['hm'].sigmoid_()
                hm = output['hm']
                # hm = _sigmoid(output['hm'])
                reg = output['reg'] if self.opt.reg_offset else None
                wh = output['wh']
                if self.opt.dataset == 'dota':
                    a = output['a']

                if self.opt.flip_test:
                    hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                    wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                    reg = reg[0:1] if reg is not None else None
                    a = a[0:1] if a is not None else None
                    # a = (a[0:1] - flip_tensor(a[1:2])) / 2. if a is not None else None
                torch.cuda.synchronize()
                forward_time = time.time()
                if self.opt.debug:
                    pred_hm = output['hm'].cpu().numpy()
                    gt_hm = hm.cpu().numpy()
                    pred_wh = output['wh'].cpu().numpy()
                    gt_wh = wh.cpu().numpy()
                det = ctdet_decode(
                    hm, wh, reg=reg, a=a, cat_spec_wh=self.opt.cat_spec_wh,
                    K=self.opt.K, a_method=self.opt.a_method,
                    debug=self.opt.debug
                )
                dets.append(det)

        if return_time:
            return outputs, dets, forward_time
        else:
            return outputs, dets

    def post_process(self, dets, meta, scale=1, fpn_stride=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'] / fpn_stride, meta['out_width'] / fpn_stride, self.opt.num_classes)
        if self.opt.dataset == 'dota':
            coor_len = 8
        else:
            coor_len = 4
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, coor_len + 1)
            dets[0][j][:, :coor_len] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            # if len(self.scales) > 1 or self.opt.nms:
            #     soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, -1] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, -1] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        coor_len = 4
        if self.opt.dataset == 'dota':
            coor_len = 8
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :coor_len] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, coor_len] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :coor_len], detection[i, k, -1],
                                           detection[i, k, coor_len],
                                           img_id='out_pred_{:.1f}'.format(scale), coor_len=coor_len)

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        coor_len = 4
        if self.opt.dataset == 'dota':
            coor_len = 8
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[coor_len] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:coor_len], j - 1, bbox[coor_len], img_id='ctdet', coor_len=coor_len)
        debugger.show_all_imgs(pause=self.pause)

    def save_results(self, debugger, image, results, path='./cache/debug', prefix='', genID=False):
        debugger.add_img(image, img_id='ctdet')
        coor_len = 4
        if self.opt.dataset == 'dota':
            coor_len = 8
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[coor_len] > self.opt.vis_thresh:
                    debugger.add_bbox_for_paper(bbox[:coor_len], j - 1, img_id='ctdet', coor_len=coor_len)
        debugger.save_all_imgs(path, prefix)
