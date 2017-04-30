#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import glob
import argparse
import os
from tqdm import tqdm

VOC_CLASSES = [
    'background'  ,
    'aeroplane'   ,
    'bicycle'     ,
    'bird'        ,
    'boat'        ,
    'bottle'      ,
    'bus'         ,
    'car'         ,
    'cat'         ,
    'chair'       ,
    'cow'         ,
    'diningtable' ,
    'dog'         ,
    'horse'       ,
    'motorbike'   ,
    'person'      ,
    'pottedplant' ,
    'sheep'       ,
    'sofa'        ,
    'train'       ,
    'tvmonitor'   ,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='path to predictions')
    parser.add_argument('--gt', required=True, help='path to groundtruths')

    parser.add_argument('--classes', type=int, default=21, help='number of classes')
    parser.add_argument('--ignore', type=int, default=255, help='number of classes')
    args = parser.parse_args()

    gtlist = glob.glob(os.path.join(args.gt, '*.png'))
    predlist = glob.glob(os.path.join(args.pred, '*.png'))
    imgs = {}
    for imgpath in predlist:
        fn = os.path.basename(imgpath)
        imgs[fn] = imgpath

    t_pos_sum = [0] * args.classes
    f_pos_sum = [0] * args.classes
    f_neg_sum = [0] * args.classes
    for imgpath in tqdm(gtlist):
        gt = cv2.imread(imgpath)
        fn = os.path.basename(imgpath)
        if fn in imgs:
            pred = cv2.imread(imgs[fn])
        else:
            pred = np.ones_like(gt, dtype=np.uint8) * args.classes

        for i in range(args.classes):
            gt_i = gt == i
            pred_i = pred == i
            gt_ni = np.logical_not(gt_i)
            pred_ni = np.logical_not(pred_i)
            t_pos = np.logical_and(gt_i, pred_i).sum()
            f_pos = np.logical_and(np.logical_and(gt_ni, gt != args.ignore), pred_i).sum()
            f_neg = np.logical_and(gt_i, pred_ni).sum()
            t_pos_sum[i] += t_pos
            f_pos_sum[i] += f_pos
            f_neg_sum[i] += f_neg

    accs = []
    for i in range(args.classes):
        acc = 100. * t_pos_sum[i] / (t_pos_sum[i] + f_pos_sum[i] + f_neg_sum[i])
        accs.append(acc)
        print('{}: {}'.format(VOC_CLASSES[i], acc))

    print('mAP:', sum(accs) / args.classes)


if __name__ == '__main__':
    main()
