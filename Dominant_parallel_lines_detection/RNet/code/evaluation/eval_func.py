import torch

import numpy as np

from sklearn.metrics import auc


def AA(x, y, threshold):
    index = np.searchsorted(x, threshold)
    x = np.concatenate([x[:index], [threshold]])
    y = np.concatenate([y[:index], [threshold]])
    return ((x[1:] - x[:-1]) * y[:-1]).sum() / threshold


def LP_distance(pt, a, b, c):
    # line - point distance eq
    dist = np.abs(a * pt[0] + b * pt[1] + c) / np.sqrt(a * a + b * b)

    return dist

def compute_error(pred, gt):
    # pred : primary line (rank 1)
    # label : gt point

    err = []
    for i in range(pred.shape[0]):
        l_a = (pred[i, 1] - pred[i, 3]) / (pred[i, 0] - pred[i, 2])
        l_b = -1 * l_a * pred[i, 0] + pred[i, 1]

        if (pred[i, 0] - pred[i, 2]) == 0:
            dist = np.abs(gt[0] - pred[i, 0])
        else:
            dist = LP_distance(gt, l_a, -1, l_b)

        err.append(dist)
    err = np.float32(err)
    return err

def calculate_AUC(err, tau):

    num = 200

    thresds = np.float32(np.linspace(0, tau, num + 1))

    result = np.zeros((num + 1), dtype=np.float32)

    for i in range(thresds.shape[0]):

        thresd = thresds[i]

        correct = 0
        error = 0

        for j in range(len(err)):
            is_correct = (err[j] < thresd)
            correct += float(np.sum(is_correct))
            error += float(np.sum(is_correct == 0))

        if (correct + error) != 0:
            result[i] = correct / (correct + error)

    AUC = auc(thresds, result) / tau

    print('AUC : %f' % AUC)

    return AUC