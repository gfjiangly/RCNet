# -*- encoding:utf-8 -*-
# @Time    : 2019/10/30 14:16
# @Author  : jiang
# @Site    : 
# @File    : test_images.py
# @Software: PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import torch
from logger import Logger

from opts import opts
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
import os.path as osp
import cvtools
import poly_nms


class Detection(object):

    def __init__(self, opt):
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        print(opt)
        Logger(opt)
        self.dataset = dataset_factory[opt.dataset]
        self.opt = opts().update_dataset_info_and_set_heads(opt, self.dataset)
        self.detector = detector_factory[opt.task](self.opt)

    def detect(self,
               img,
               det_thrs=0.5,
               vis=False,
               vis_thr=0.5,
               save_root=''):
        result = self.detector.run(img)
        self.vis_dets(img, result)
        return result

    def vis_dets(self, img_name, result):
        dets = result['results']
        img = cvtools.imread(img_name)
        for cls_id, det_cls in dets.items():
            det_cls = det_cls[det_cls[:, -1] > 0.5]
            if len(det_cls) == 0: continue
            ids = poly_nms.poly_gpu_nms(det_cls, 0.15)
            det_cls = det_cls[ids]
            if cls_id == 15:
                img = cvtools.draw_boxes_texts(img, det_cls[:, :-1],
                                            line_width=2,
                                            box_format="polygon")
        img_name = osp.basename(img_name)
        to_file = osp.join('/code/CenterNet/exp/ctdet/vis', img_name)
        cvtools.imwrite(img, to_file)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    opt = opts().parse()
    detector = Detection(opt)
    # detector.detect(r'/media/data/DOTA/crop/P2802_4095_2457_5118_3480.png')
    # detector.detect(r'/media/data/DOTA/crop/P2701_2880_1440_3679_2239.png')  # 存储罐
    # detector.detect(r'/media/data/DOTA/crop/P2082_0_0_799_799.png')  # 车
    # detector.detect(r'/media/data/DOTA/crop/P0377_0_0_799_799.png')  # 网球场
    # detector.detect(r'/media/data/DOTA/crop/P2780_1025_720_1825_1520.png')  # 飞机
    # detector.detect(r'/media/data/DOTA/crop/P2736_0_0_800_800.png')  # 存储罐2
    # detector.detect(r'/media/data/DOTA/crop/P0337_824_0_1847_1023.png')  # 港口+泳池
    detector.detect(r'/media/data/DOTA/crop/P0158_1678_270_2701_1293.png')  # 圆环
