import torch

import numpy as np
import matplotlib.pyplot as plt

def measure_IoU(X1, X2):
    X = X1 + X2

    X_uni = torch.sum(X != 0, dim=(1, 2)).type(torch.float32)
    X_inter = torch.sum(X == 2, dim=(1, 2)).type(torch.float32)

    iou = X_inter / X_uni

    check = (X_inter > 0)

    return iou, check

def non_maximum_suppression(score, mask, thresd):

    num = mask.shape[0]

    sorted_idx = torch.argsort(score, descending=True)

    visit = torch.zeros((num), dtype=torch.float32).cuda()
    nms_check_all = torch.zeros((num), dtype=torch.float32).cuda()
    for i in range(num):

        ref_idx = sorted_idx[i]

        if visit[ref_idx] == 1 or nms_check_all[ref_idx] == 1:
            continue

        visit[ref_idx] = 1
        max_miou = torch.zeros((num), dtype=torch.float32).cuda()

        iou_1, check1 = measure_IoU(mask[ref_idx, 0].unsqueeze(0), mask[:, 0])
        iou_2, check2 = measure_IoU(mask[ref_idx, 1].unsqueeze(0), mask[:, 1])

        check = (check1 * check2).type(torch.float32)
        max_check = (max_miou < check * (iou_1 + iou_2) / 2).type(torch.float32)
        max_miou[max_check == 1] = (check * (iou_1 + iou_2) / 2)[max_check == 1]

        iou_1, check1 = measure_IoU(mask[ref_idx, 1].unsqueeze(0), mask[:, 0])
        iou_2, check2 = measure_IoU(mask[ref_idx, 0].unsqueeze(0), mask[:, 1])

        check = (check1 * check2).type(torch.float32)
        max_check = (max_miou < check * (iou_1 + iou_2) / 2).type(torch.float32)
        max_miou[max_check == 1] = (check * (iou_1 + iou_2) / 2)[max_check == 1]

        nms_check = (max_miou > thresd) * (max_miou < 1) * (visit == 0)
        nms_check_all[nms_check == 1] = 1

    return nms_check_all

def divided_region_mask(line_pts, size):

    line_num, _ = line_pts.shape
    width, height = int(size[0]), int(size[1])

    X, Y = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height))  # after x before
    X = torch.tensor(X, dtype=torch.float, requires_grad=False).unsqueeze(0).cuda()
    Y = torch.tensor(Y, dtype=torch.float, requires_grad=False).unsqueeze(0).cuda()

    check = ((line_pts[:, 0] - line_pts[:, 2]) == 0).type(torch.float)

    mask1 = torch.zeros((line_num, height, width), dtype=torch.float32).cuda()
    mask2 = torch.zeros((line_num, height, width), dtype = torch.float32).cuda()

    mask1[check == 1, :, :] = (X < line_pts[:, 0].view(line_num, 1, 1)).type(torch.float)[check == 1, :, :]
    mask2[check == 1, :, :] = (X >= line_pts[:, 0].view(line_num, 1, 1)).type(torch.float)[check == 1, :, :]

    a = (line_pts[:, 1] - line_pts[:, 3]) / (line_pts[:, 0] - line_pts[:, 2])
    b = -1 * a * line_pts[:, 0] + line_pts[:, 1]

    a = a.view(line_num, 1, 1)
    b = b.view(line_num, 1, 1)

    mask1[check == 0, :, :] = (Y < a * X + b).type(torch.float32)[check == 0, :, :]
    mask2[check == 0, :, :] = (Y >= a * X + b).type(torch.float32)[check == 0, :, :]

    return torch.cat((mask1.unsqueeze(1), mask2.unsqueeze(1)), dim=1)


def find_endpoints(data, size):
    x1, y1, x2, y2 = data[0], data[1], data[2], data[3]

    pts = []
    if x1 - x2 != 0:
        a = (y1 - y2) / (x1 - x2)
        b = -1
        c = -1 * a * x1 + y1
        # x = 0
        cx = 0
        cy = a * 0 + c
        if cy >= 0 and cy <= size[1]:
            pts.append(cx)
            pts.append(cy)
        # x = size[0]
        cx = size[0]
        cy = a * size[0] + c
        if cy >= 0 and cy <= size[1]:
            pts.append(cx)
            pts.append(cy)
        # y = 0
        if y1 != y2:
            cx = (0 - c) / a
            cy = 0
            if cx >= 0 and cx <= size[0]:
                pts.append(cx)
                pts.append(cy)
            # y = size[1]
            cx = (size[1] - c) / a
            cy = size[1]
            if cx >= 0 and cx <= size[0]:
                pts.append(cx)
                pts.append(cy)
    else:
        if x1 >= 0 and x1 <= size[0]:
            pts.append(x1)
            pts.append(0)
            pts.append(x1)
            pts.append(size[1])

    return np.float32(pts)


def suppression(top_m, mask, thresd):

    num = mask.shape[0]

    ref_idx = top_m

    max_miou = torch.zeros((num), dtype=torch.float32).cuda()

    iou_1, check1 = measure_IoU(mask[ref_idx, 0].unsqueeze(0), mask[:, 0])
    iou_2, check2 = measure_IoU(mask[ref_idx, 1].unsqueeze(0), mask[:, 1])

    check = (check1 * check2).type(torch.float32)
    max_check = (max_miou < check * (iou_1 + iou_2) / 2).type(torch.float32)
    max_miou[max_check == 1] = (check * (iou_1 + iou_2) / 2)[max_check == 1]

    iou_1, check1 = measure_IoU(mask[ref_idx, 1].unsqueeze(0), mask[:, 0])
    iou_2, check2 = measure_IoU(mask[ref_idx, 0].unsqueeze(0), mask[:, 1])

    check = (check1 * check2).type(torch.float32)
    max_check = (max_miou < check * (iou_1 + iou_2) / 2).type(torch.float32)
    max_miou[max_check == 1] = (check * (iou_1 + iou_2) / 2)[max_check == 1]

    check = (max_miou > thresd) * (max_miou < 1)

    return check