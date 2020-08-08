import random
import torch

import numpy as np

from libs.utils import *

def generate_line_pair_RNet(pos_line, neg_line, line_batch_size, mode='training'):
    # positive line
    pos_line = pos_line['pos_line']
    gt_num = len(pos_line)

    gt_list = []
    for i in range(gt_num):
         gt_list.append(i)

    # negative line
    if mode == 'training':
        neg_line['outlier_line'][0] = neg_line['outlier_line'][0][0, :]
        for i in range(gt_num):
            pos_line[i] = pos_line[i][0, :]
            neg_line['inlier_line'][i] = neg_line['inlier_line'][i][0, :]


    # reference, target, label
    ref = np.zeros((line_batch_size * 4, 4), dtype=np.float32)
    tar = np.zeros((line_batch_size * 4, 4), dtype=np.float32)
    label = np.zeros((line_batch_size * 4, 2), dtype=np.float32)

    k = 0

    for case in range(2):

        for i in range(line_batch_size):

            if case == 0:
                # reference line is more semantic than target line
                idx_gt_line = np.random.randint(0, gt_num)

                line1 = pos_line[idx_gt_line]
                line2 = neg_line['inlier_line'][idx_gt_line]

                num1 = line1.shape[0]
                num2 = line2.shape[0]

                if num2 == 0:
                    line2 = neg_line['outlier_line'][0]
                    num2 = line2.shape[0]

                idx_line1 = np.random.randint(0, num1)
                idx_line2 = np.random.randint(0, num2)

                ref[k, :] = line1[idx_line1, :4]
                tar[k, :] = line2[idx_line2, :4]

            elif case == 1:
                # reference line is less semantic than target line
                idx_gt_line = np.random.randint(0, gt_num)

                line1 = pos_line[idx_gt_line]
                line2 = neg_line['inlier_line'][idx_gt_line]

                num1 = line1.shape[0]
                num2 = line2.shape[0]

                if num2 == 0:
                    line2 = neg_line['outlier_line'][0]
                    num2 = line2.shape[0]

                idx_line1 = np.random.randint(0, num1)
                idx_line2 = np.random.randint(0, num2)

                ref[k, :] = line2[idx_line2, :4]
                tar[k, :] = line1[idx_line1, :4]

            # label
            label[k, case] = 1
            k += 1

    ref = ref[:k, :]
    tar = tar[:k, :]
    label = label[:k, :]

    if mode == 'training':
        idx = np.arange(k)
        np.random.shuffle(idx)
        ref = to_tensor(ref[idx, :]).unsqueeze(0)
        tar = to_tensor(tar[idx, :]).unsqueeze(0)
        label = to_tensor(label[idx, :]).unsqueeze(0)

    return {'ref': ref, 'tar': tar, 'label': label, 'num': k}
