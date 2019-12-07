# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:43:43 2019

@author: tanma
"""

import re
import cv2

splitter = re.compile("\s+")

# Get and Create Bounding Boxes
def create_dict_bboxes(list_all, split='train'):
    lst = [(line[0], line[1], line[3], line[2]) for line in list_all if line[2] == split]
    lst = [("".join(line[0].split('/')[0] + '/' + line[3] + '/' + line[1] + line[0][3:]), line[1], line[2]) for line in lst]
    lst_shape = [cv2.imread('./data/' + line[0]).shape for line in lst]
    lst = [(line[0], line[1], (round(line[2][0] / shape[1], 2), round(line[2][1] / shape[0], 2), round(line[2][2] / shape[1], 2), round(line[2][3] / shape[0], 2))) for line, shape in zip(lst, lst_shape)]
    dict_ = {"/".join(line[0].split('/')[2:]): {'x1': line[2][0], 'y1': line[2][1], 'x2': line[2][2], 'y2': line[2][3], 'shape': line[2][4]} for line in lst}
    return dict_

def get_dict_bboxes():
    with open('./data/anno/list_category_img.txt', 'r') as category_img_file, \
            open('./data/anno/list_eval_partition.txt', 'r') as eval_partition_file, \
            open('./data/anno/list_bbox.txt', 'r') as bbox_file:
        list_category_img = [line.rstrip('\n') for line in category_img_file][2:]
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_bbox = [line.rstrip('\n') for line in bbox_file][2:]

        list_category_img = [splitter.split(line) for line in list_category_img]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        list_bbox = [splitter.split(line) for line in list_bbox]

        list_all = [(k[0], k[0].split('/')[1].split('_')[-1], v[1], (int(b[1]), int(b[2]), int(b[3]), int(b[4])))
                    for k, v, b in zip(list_category_img, list_eval_partition, list_bbox)]

        list_all.sort(key=lambda x: x[1])

        dict_train = create_dict_bboxes(list_all)
        dict_val = create_dict_bboxes(list_all, split='val')
        dict_test = create_dict_bboxes(list_all, split='test')

        return dict_train, dict_val, dict_test