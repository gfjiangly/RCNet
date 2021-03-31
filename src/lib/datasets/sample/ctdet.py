# coding: utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import cv2
import os
import os.path as osp
import math
import random
from shapely.geometry import Polygon
import copy
import cvtools
from cvtools.utils.boxes import x1y1wh_to_x1y1x2y2x3y3x4y4

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg


def convert_angle(a, w, h, method=0):
    if 0. >= a >= -90.:
        if method == 0:
            a = -a
        elif method == 1 or method == 2:
            if w < h:
                a = 90. + a
                w, h = h, w
    elif a == 90.:
        a = 0.
        w, h = h, w
    else:
        a = 0.
    return a, w, h


# def convert_angle(a, w, h, method=0):
#     if 0. > a >= -90.:
#         if method == 0:
#             a = -a
#         elif method == 1 or method == 2:
#             if w < h:
#                 a = 90. + a
#                 w, h = h, w
#     elif a == 90.:
#         a = 0.
#         w, h = h, w
#     else:
#         a = 0.
#     return a, w, h


def cut_polygon(polygon, box):
    # polygon可以不按点的顺序构建，但是必须使用convex_hull求交
    polygon = Polygon(polygon).convex_hull
    box = Polygon(box).convex_hull
    # boundary属性对应的LineString可以直接转为array对象
    # 最后一个点与第一个点相同，予以去除
    inters = polygon.intersection(box)
    try:
        intersections = np.array(inters.boundary, dtype=np.float)[:-1]
        bounds = np.array(inters.bounds, dtype=np.float)    # 最小外接矩形
    except ValueError:
        return None, None
    return intersections, bounds


def re_anns(anns, trans_output, output_w, output_h):
    # gt_boxes = [ann['bbox'] for ann in anns]
    # gt_boxes = cvtools.x1y1wh_to_x1y1x2y2(np.array(gt_boxes, dtype=np.float))
    # for bbox in gt_boxes:
    #     bbox[:2] = affine_transform(bbox[:2], trans_output)
    #     bbox[2:] = affine_transform(bbox[2:], trans_output)
    # iofs = cvtools.bbox_overlaps(
    #     gt_boxes, np.array([[0, 0, output_w - 1, output_h - 1]]), mode='iof'
    # )
    # ids = np.where(iofs > 0.7)[0]
    # anns = [anns[ind] for ind in ids]
    # num_objs = len(ids)
    # iofs = iofs[ids]
    for k in range(len(anns)):
        # segm = np.array(anns[k]['segmentation'][0])
        segm = anns[k]['segmentation'][0]
        for i in range(0, len(segm), 2):
            segm[i:i + 2] = affine_transform(segm[i:i + 2], trans_output)
            # segm[i] = np.clip(segm[i], 0, output_w - 1)
            # segm[i + 1] = np.clip(segm[i + 1], 0, output_h - 1)
        # if iofs[k] < 1.:
        #     img_box_polygon = np.array(
        #         x1y1wh_to_x1y1x2y2x3y3x4y4(
        #             [0, 0, output_w, output_h])
        #     ).reshape(-1, 2)
        #     segm_cutted, _ = cut_polygon(
        #         segm.reshape(-1, 2), img_box_polygon
        #     )
        #     assert segm_cutted is not None and len(segm_cutted) > 0
        #     segm = segm_cutted
        # anns[k]['segmentation'] = [segm.reshape(-1).tolist()]
    return anns


class CTDetDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _get_dota_item(self, index):
        img_id = self.images[index]
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']

        filename = osp.splitext(file_name)[0]
        suffix = osp.splitext(file_name)[1]
        crop_str = list(map(str, img_info['crop']))
        crop_img_path = osp.join('/code/data/DOTA/crop800_80',
                                 '_'.join([filename] + crop_str) + suffix)
        if not osp.isfile(crop_img_path):
            img_path = os.path.join('/media/data/DOTA/trainval/images', file_name)
            img = cv2.imread(img_path)
            sx, sy, ex, ey = img_info['crop']
            img = img[sy:ey+1, sx:ex+1]
            cv2.imwrite(crop_img_path, img)
        else:
            img = cv2.imread(crop_img_path)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        # 如果不deep拷贝，修改了anns就修改了self.coco里的标签
        anns = copy.deepcopy(self.coco.loadAnns(ids=ann_ids))

        # if True:
        if self.opt.debug:
            segs = [ann['segmentation'][0] for ann in anns]
            cvtools.imwrite(
                cvtools.draw_boxes_texts(img.copy(), segs, box_format='polygon'),
                self.opt.debug_dir + '/{}'.format(file_name))

        if self.opt.flip:
            try:
                hv_flip = cvtools.RandomMirror(both=False)
                segs = [ann['segmentation'][0].copy() for ann in anns]
                for i, seg in enumerate(segs):
                    if len(seg) != 8:
                        segm_hull = cv2.convexHull(
                            np.array(seg).reshape(-1, 2).astype(np.float32),
                            clockwise=False)
                        xywha = cv2.minAreaRect(segm_hull)
                        segs[i] = cv2.boxPoints(xywha).reshape(-1).tolist()
                img, segs = hv_flip(img, segs)
                # if True:
                if self.opt.debug:
                    cvtools.imwrite(
                        cvtools.draw_boxes_texts(img.copy(), segs,
                                                 box_format='polygon'),
                        self.opt.debug_dir + '/flip_{}'.format(file_name)
                    )
                for i in range(len(anns)):
                    anns[i]['segmentation'][0] = list(segs[i])
                    bbox = cv2.boundingRect(np.array(segs[i], dtype=np.float32).reshape(-1, 2))
                    anns[i]['bbox'] = list(bbox)
            except Exception as e:
                print(e)
                return []

        if self.opt.rotate:
            rotate = cvtools.RandomRotate()
            segs = [ann['segmentation'][0].copy() for ann in anns]
            for i, seg in enumerate(segs):
                if len(seg) != 8:
                    segm_hull = cv2.convexHull(
                        np.array(seg).reshape(-1, 2).astype(np.float32),
                        clockwise=False)
                    xywha = cv2.minAreaRect(segm_hull)
                    segs[i] = cv2.boxPoints(xywha).reshape(-1).tolist()
            img, segs = rotate(img, segs)
            # if True:
            if self.opt.debug:
                cvtools.imwrite(
                    cvtools.draw_boxes_texts(img.copy(), segs, box_format='polygon'),
                    self.opt.debug_dir + '/rotate_{}'.format(file_name)
                )
            for i in range(len(anns)):
                anns[i]['segmentation'][0] = list(segs[i])
                bbox = cv2.boundingRect(np.array(segs[i], dtype=np.float32).reshape(-1, 2))
                anns[i]['bbox'] = list(bbox)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        # self.opt.input_h = self.opt.input_w = 32 * random.randint(12, 20)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        # flipped = False
        if 'train' in self.split:
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            # if np.random.random() < self.opt.flip:
            #     flipped = True
            #     img = img[:, ::-1, :]
            #     c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)

        gt_boxes = np.array(
            [cvtools.x1y1wh_to_x1y1x2y2(ann['bbox']) for ann in anns]
        )
        img_box = cvtools.xywh_to_x1y1x2y2(np.array([[c[0], c[1], s, s]]))
        img_box[0, 0::2] = np.clip(img_box[0, 0::2], 0, width - 1)
        img_box[0, 1::2] = np.clip(img_box[0, 1::2], 0, height - 1)
        iofs = cvtools.bbox_overlaps(gt_boxes, img_box, mode='iof')
        ids = np.where(iofs > 0.7)[0]
        if len(ids) == 0: return []
        anns = [anns[ind] for ind in ids]

        # if True:
        if self.opt.debug:
            segs = [ann['segmentation'][0].copy() for ann in anns]  # 复制一份，否则是原视图
            inp_draw = inp.copy()
            for k in range(len(segs)):
                seg = segs[k]
                for i in range(0, len(seg), 2):
                    seg[i:i + 2] = affine_transform(seg[i:i + 2], trans_input)
                    # seg[i] = np.clip(seg[i], 0, input_w - 1)
                    # seg[i + 1] = np.clip(seg[i + 1], 0, input_h - 1)
                segm_hull = cv2.convexHull(
                    np.array(seg).reshape(-1, 2).astype(np.float32), clockwise=False
                )
                xy, _, _ = cv2.minAreaRect(segm_hull)
                cv2.circle(inp_draw, (int(xy[0]), int(xy[1])),
                           radius=5, color=(0, 0, 255), thickness=-1)
            cvtools.imwrite(
                cvtools.draw_boxes_texts(
                    inp_draw, segs, draw_start=False, box_format='polygon'),
                osp.join(self.opt.debug_dir, 'trans_'+file_name)
            )

        inp = (inp.astype(np.float32) / 255.)
        if 'train' in self.split and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        rets = []
        # out_size = []
        # for down_ratio in down_ratios:
        #     output_h = input_h // down_ratio
        #     output_w = input_w // down_ratio
        #     num_classes = self.num_classes
        #     out_size.append([output_w, output_h])
        #     # trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
        #
        #     hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        #     wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        #     if self.opt.a_method == 2:
        #         angle = np.full((self.max_objs, 1), 0.5, dtype=np.float32)
        #     else:
        #         angle = np.zeros((self.max_objs, 1), dtype=np.float32)
        #     dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        #     reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        #     ind = np.zeros((self.max_objs), dtype=np.int64)
        #     reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        #     cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        #     cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
        #
        #     # for k in range(num_objs):
        #     #     cls_id = int(self.cat_ids[anns[k]['category_id']])
        #     #     draw_heatmap(hm[cls_id], osp.join(self.opt.debug_dir, 'heatmap_' + str(cls_id) + '_' + file_name))
        #     ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'a': angle}
        #     if self.opt.dense_wh:
        #         hm_a = hm.max(axis=0, keepdims=True)
        #         dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
        #         ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
        #         del ret['wh']
        #     elif self.opt.cat_spec_wh:
        #         ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
        #         del ret['wh']
        #     if self.opt.reg_offset:
        #         ret.update({'reg': reg})
        #     # if self.opt.debug > 0 or not self.split == 'train':
        #     #     gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
        #     #         np.zeros((1, 6), dtype=np.float32)
        #     #     meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
        #     #     ret['meta'] = meta
        #     rets.append(ret)
        #
        # draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
        #     draw_umich_gaussian
        # if not self.opt.fpn:
        #     output_w, output_h = out_size[0]
        #     trans_output = get_affine_transform(c, s, 0, out_size[0])
        #
        # for k in range(num_objs):
        #     ann = anns[k]
        #     cls_id = int(self.cat_ids[ann['category_id']])
        #
        #     # 确定GT分配给哪个FPN层
        #     if self.opt.fpn:
        #         bbox = ann['bbox']
        #         fpn_k = int(math.log(224. / math.sqrt(bbox[2] * bbox[3]), 2))
        #         if fpn_k < 0:
        #             fpn_k = 0
        #         if fpn_k > 2:
        #             fpn_k = 2
        #         ret = rets[fpn_k]
        #         output_w, output_h = out_size[fpn_k]
        #         trans_output = get_affine_transform(c, s, 0, out_size[fpn_k])
        #
        #     segm = np.array(ann['segmentation'][0])
        #     # if flipped:
        #     #     for i in range(0, len(segm), 2):
        #     #         segm[i] = width - segm[i] - 1
        #     for i in range(0, len(segm), 2):
        #         segm[i:i + 2] = affine_transform(segm[i:i + 2], trans_output)
        #         segm[i] = np.clip(segm[i], 0, output_w - 1)
        #         segm[i + 1] = np.clip(segm[i + 1], 0, output_h - 1)
        #
        #     segm_hull = cv2.convexHull(segm.reshape(-1, 2).astype(np.float32),
        #                                clockwise=False)
        #     xy, (w, h), a = cv2.minAreaRect(segm_hull)
        #     hm = ret['hm']
        #     reg_mask = ret['reg_mask']
        #     ind = ret['ind']
        #     wh = ret['wh']
        #     angle = ret['a']
        #     if h > 0 and w > 0:
        #         a, w, h = convert_angle(a, w, h, self.opt.a_method)
        #         ct = np.array(xy, dtype=np.float32)
        #         ct_int = ct.astype(np.int32)
        #         radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        #         radius = max(0, int(radius))
        #         radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        #         # radius = np.array((h / 3., w / 3.), np.int32)
        #         draw_gaussian(hm[cls_id], ct_int, radius)
        #         wh[k] = 1. * w, 1. * h
        #         gt_a = a / 90.
        #         if self.opt.a_method == 2:
        #             gt_a = (a + 90.) / 180.
        #         angle[k] = gt_a
        #         ind[k] = ct_int[1] * output_w + ct_int[0]
        #         if 'reg' in ret:
        #             ret['reg'][k] = ct - ct_int
        #         reg_mask[k] = 1
        #         if 'cat_spec_wh' in ret:
        #             ret['cat_spec_wh'][k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        #         if 'cat_spec_mask' in ret:
        #             ret['cat_spec_mask'][k, cls_id * 2: cls_id * 2 + 2] = 1
        #         if self.opt.dense_wh:
        #             draw_dense_reg(ret['dense_wh'], hm.max(axis=0), ct_int,
        #                            wh[k], radius)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        if self.opt.a_method == 2:
            angle = np.full((self.max_objs, 1), 0.5, dtype=np.float32)
        else:
            angle = np.zeros((self.max_objs, 1), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        anns = re_anns(anns, trans_output, output_w, output_h)
        num_objs = min(len(anns), self.max_objs)

        # if True:
        if self.opt.debug:
            gt_img = cv2.warpAffine(img, trans_output,
                                    (output_w, output_h),
                                    flags=cv2.INTER_LINEAR)
            segs = [ann['segmentation'][0] for ann in anns]
            cvtools.imwrite(
                cvtools.draw_boxes_texts(
                    gt_img, segs, draw_start=False, box_format='polygon'),
                osp.join(self.opt.debug_dir, 'gt_'+file_name)
            )

        bad_num = 0
        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(self.cat_ids[ann['category_id']])
            segm = np.array(ann['segmentation'][0])
            # if flipped:
            #     for i in range(0, len(segm), 2):
            #         segm[i] = width - segm[i] - 1
            # for i in range(0, len(segm), 2):
            #     segm[i:i + 2] = affine_transform(segm[i:i + 2], trans_output)
            #     segm[i] = np.clip(segm[i], 0, output_w - 1)
            #     segm[i + 1] = np.clip(segm[i + 1], 0, output_h - 1)

            segm_hull = cv2.convexHull(
                segm.reshape(-1, 2).astype(np.float32), clockwise=False
            )
            xy, (w, h), a = cv2.minAreaRect(segm_hull)
            if xy[0] > output_w or xy[0] < 0 or xy[1] > output_h or xy[1] < 0:
                # TODO：查明为何会出现这种情况。P0750
                # xy中y下出现负值或大于127
                # print(file_name, ann, segm, xy)
                bad_num += 1
                continue
            if h > 0 and w > 0:
                a, w, h = convert_angle(a, w, h, self.opt.a_method)
                ct = np.array(xy, dtype=np.float32)
                ct_int = ct.astype(np.int32)
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                # radius = np.array((h / 3., w / 3.), np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                gt_a = a / 90.
                if self.opt.a_method == 2:
                    gt_a = (a + 90.) / 180.
                angle[k] = gt_a
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k],
                                   radius)
                gt_det.append(segm + [cls_id])
            else:
                bad_num += 1

        if bad_num == num_objs: return []
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'a': angle}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or 'train' not in self.split:
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        rets.append(ret)
        return rets

    def __getitem__(self, index):
        if self.opt.dataset == 'dota':
            rets = self._get_dota_item(index)
            if len(rets) == 0:
                new_index = random.randint(0, len(self) - 1)
                return self[new_index]
            return rets
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret
