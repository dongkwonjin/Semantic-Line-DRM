import torch

import numpy as np

from evaluation.eval_func import *

def eval_AA(out, gt):
    num = len(out)
    err = []
    # eval process
    for i in range(num):
        err.append(min(np.arccos(np.abs(out[i] @ gt[i][0]).clip(max=1))) / np.pi * 180)

    # performance
    err = np.sort(np.array(err))
    y = (1 + np.arange(len(err))) / len(err)
    print(" | ".join([f"{AA(err, y, th):.3f}" for th in [0.2, 0.5, 1.0, 2.0, 10.0]]))

    return [AA(err, y, th) for th in [0.2, 0.5, 1.0, 2.0, 10.0]]

def eval_dist_AUC(out, gt):
    num = len(out)
    err = []
    # eval process
    for i in range(num):
        err.append(compute_error(out[i], gt[i]))

    # performance
    auc = calculate_AUC(err, tau=10)
    return auc