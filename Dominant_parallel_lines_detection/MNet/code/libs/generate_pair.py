import random
import torch

import numpy as np

from libs.utils import *

def generate_line_pair_MNet(training_line, line_batch_size, mode='training'):

    pos_num = len(training_line['pos'])
    neg_num = len(training_line['neg'])

    if mode == 'training':
        for i in range(pos_num):
            training_line['pos'][i] = training_line['pos'][i][0, :]
        for i in range(neg_num):
            training_line['neg'][i] = training_line['neg'][i][0, :]


    # reference, target, label
    ref = np.zeros((line_batch_size * 4, 4), dtype=np.float32)
    tar = np.zeros((line_batch_size * 4, 4), dtype=np.float32)
    label = np.zeros((line_batch_size * 4, 2), dtype=np.float32)

    k = 0
    pos = []
    neg = []
    for i in range(pos_num):
        pos.append(i)
    for i in range(neg_num):
        neg.append(i)

    for case in range(2):
        for i in range(line_batch_size):

            if case == 0:
                # negative

                c = np.random.randint(0, 3)
                if c == 0:  # (pos, neg)
                    idx1 = np.random.randint(0, pos_num)
                    idx2 = np.random.randint(0, neg_num)

                    num1 = training_line['pos'][idx1].shape[0]
                    num2 = training_line['neg'][idx2].shape[0]

                    if num1 == 0 or num2 == 0:
                        continue
                    idx_line1 = np.random.randint(0, num1)
                    idx_line2 = np.random.randint(0, num2)

                    line1 = training_line['pos'][idx1][idx_line1, :4]
                    line2 = training_line['neg'][idx2][idx_line2, :4]

                elif c == 1:  # (neg, neg)
                    idx1 = np.random.randint(0, neg_num)
                    idx2 = np.random.randint(0, neg_num)

                    num1 = training_line['neg'][idx1].shape[0]
                    num2 = training_line['neg'][idx2].shape[0]

                    if num1 == 0 or num2 == 0:
                        continue
                    idx_line1 = np.random.randint(0, num1)
                    idx_line2 = np.random.randint(0, num2)

                    line1 = training_line['neg'][idx1][idx_line1, :4]
                    line2 = training_line['neg'][idx2][idx_line2, :4]
                else:  # (pos_1, pos_1)
                    idx1 = np.random.randint(0, pos_num - 1)
                    idx2 = idx1
                    num1 = training_line['pos'][idx1].shape[0]
                    num2 = training_line['pos'][idx2].shape[0]

                    if num1 == 0 or num2 == 0:
                        continue
                    idx_line1 = np.random.randint(0, num1)
                    idx_line2 = np.random.randint(0, num2)

                    line1 = training_line['pos'][idx1][idx_line1, :4]
                    line2 = training_line['pos'][idx2][idx_line2, :4]


                flip = np.random.randint(0, 2)
                if flip == 0:
                    ref[k, :] = line1
                    tar[k, :] = line2
                else:
                    ref[k, :] = line2
                    tar[k, :] = line1

            elif case == 1:
                # positive

                idx1, idx2 = random.sample(pos, 2)

                num1 = training_line['pos'][idx1].shape[0]
                num2 = training_line['pos'][idx2].shape[0]

                if num1 == 0 or num2 == 0:
                    continue
                idx_line1 = np.random.randint(0, num1)
                idx_line2 = np.random.randint(0, num2)

                ref[k, :] = training_line['pos'][idx1][idx_line1, :4]
                tar[k, :] = training_line['pos'][idx2][idx_line2, :4]


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