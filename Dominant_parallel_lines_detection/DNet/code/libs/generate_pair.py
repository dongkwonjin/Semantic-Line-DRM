import random
import torch

import numpy as np

from libs.utils import *

def generate_line_pair_RNet(training_line, line_batch_size, mode='training'):

    num = len(training_line['data'])

    if mode == 'training':
        for i in range(num):
            training_line['data'][i] = training_line['data'][i][0, :]


    # reference, target, label
    ref = np.zeros((line_batch_size * 4, 4), dtype=np.float32)
    tar = np.zeros((line_batch_size * 4, 4), dtype=np.float32)
    label = np.zeros((line_batch_size * 4, 2), dtype=np.float32)

    k = 0

    for case in range(2):
        for i in range(line_batch_size):

            if case == 0:
                # reference line is more aligned with vanishing point than target line

                idx_rank1 = np.random.randint(0, num-1)
                idx_rank2 = np.random.randint(idx_rank1 + 1, num)

                num1 = training_line['data'][idx_rank1].shape[0]
                num2 = training_line['data'][idx_rank2].shape[0]

                if num1 == 0 or num2 == 0:
                    continue

                idx_line1 = np.random.randint(0, num1)
                idx_line2 = np.random.randint(0, num2)

                ref[k, :] = training_line['data'][idx_rank1][idx_line1, :4]
                tar[k, :] = training_line['data'][idx_rank2][idx_line2, :4]

            elif case == 1:
                # target line is more aligned with vanishing point than reference line
                idx_rank1 = np.random.randint(0, num-1)
                idx_rank2 = np.random.randint(idx_rank1 + 1, num)

                num1 = training_line['data'][idx_rank1].shape[0]
                num2 = training_line['data'][idx_rank2].shape[0]

                if num1 == 0 or num2 == 0:
                    continue

                idx_line1 = np.random.randint(0, num1)
                idx_line2 = np.random.randint(0, num2)

                tar[k, :] = training_line['data'][idx_rank1][idx_line1, :4]
                ref[k, :] = training_line['data'][idx_rank2][idx_line2, :4]


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