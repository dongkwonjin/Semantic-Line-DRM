import os
import pickle

import torch

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc

class Evaluation_Function(object):

    def __init__(self, cfg=None):

        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.X, self.Y = np.meshgrid(np.linspace(0, self.width - 1, self.width),
                                     np.linspace(0, self.height - 1, self.height))
        self.X = torch.tensor(self.X, dtype=torch.float, requires_grad=False).cuda()
        self.Y = torch.tensor(self.Y, dtype=torch.float, requires_grad=False).cuda()

    def measure_miou(self, out, gt):


        ### mask
        output_mask = divided_region_mask(line_pts=out,
                                          size=[self.width, self.height])
        gt_mask = divided_region_mask(line_pts=gt,
                                      size=[self.width, self.height])

        # miou
        precision_miou = measure_IoU_set(ref_mask=gt_mask,
                                         tar_mask=output_mask)
        recall_miou = precision_miou.clone().permute(1, 0).contiguous()

        return precision_miou, recall_miou

    def matching(self, miou, idx):

        out_num, gt_num = miou['p'][idx].shape

        precision = torch.zeros((out_num), dtype=torch.float32).cuda()
        recall = torch.zeros((gt_num), dtype=torch.float32).cuda()

        for i in range(out_num):
            if gt_num == 0:
                break

            max_idx = torch.argmax(miou['p'][idx].view(-1))

            if miou['p'][idx].view(-1)[max_idx] == -1:
                continue

            out_idx = max_idx / gt_num
            gt_idx = max_idx % gt_num

            precision[out_idx] = miou['p'][idx].view(-1)[max_idx]
            miou['p'][idx][out_idx, :] = -1
            miou['p'][idx][:, gt_idx] = -1

        for i in range(gt_num):
            if out_num == 0:
                break

            max_idx = torch.argmax(miou['r'][idx].view(-1))

            if miou['r'][idx].view(-1)[max_idx] == -1:
                continue

            gt_idx = max_idx / out_num
            out_idx = max_idx % out_num

            recall[gt_idx] = miou['r'][idx].view(-1)[max_idx]
            miou['r'][idx][gt_idx, :] = -1
            miou['r'][idx][:, out_idx] = -1

        return precision, recall


    def calculate_AUC(self, miou, metric):

        num = 200

        thresds = np.float32(np.linspace(0, 1, num + 1))

        result = torch.zeros((num + 1), dtype=torch.float32)

        for i in range(thresds.shape[0]):

            thresd = thresds[i]

            correct = 0
            error = 0

            for j in miou[metric[0]]:

                if miou[metric[0]][j].shape[0] != 0:
                    is_correct = (miou[metric[0]][j] > thresd)
                    correct += float(torch.sum(is_correct))
                    error += float(torch.sum(is_correct == 0))


            if (correct + error) != 0:
                result[i] = correct / (correct + error)

        result = result.cpu().numpy()

        AUC = auc(thresds[10:191], result[10:191]) / 0.9
        print('AUC_ %s : %f' % (metric, AUC))

        if self.cfg.draw_auc_graph == True:

            data = []
            data.append(thresds)
            data.append(result)
            data.append(metric)

            draw_plot(data, self.cfg.output_dir + 'test/')

        return AUC


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

def measure_IoU_set(ref_mask, tar_mask):
    ref_num = ref_mask.shape[0]
    tar_num = tar_mask.shape[0]

    miou = torch.zeros((tar_num, ref_num), dtype=torch.float32).cuda()

    for i in range(tar_num):
        iou_1, check1 = measure_IoU(tar_mask[i, 0].unsqueeze(0), ref_mask[:, 0])
        iou_2, check2 = measure_IoU(tar_mask[i, 1].unsqueeze(0), ref_mask[:, 1])

        check = (check1 * check2).type(torch.float32)
        max_check = (miou[i] < check * (iou_1 + iou_2) / 2).type(torch.float32)
        miou[i][max_check == 1] = (check * (iou_1 + iou_2) / 2)[max_check == 1]

        iou_1, check1 = measure_IoU(tar_mask[i, 1].unsqueeze(0), ref_mask[:, 0])
        iou_2, check2 = measure_IoU(tar_mask[i, 0].unsqueeze(0), ref_mask[:, 1])

        check = (check1 * check2).type(torch.float32)
        max_check = (miou[i] < check * (iou_1 + iou_2) / 2).type(torch.float32)
        miou[i][max_check == 1] = (check * (iou_1 + iou_2) / 2)[max_check == 1]

    return miou

def measure_IoU(X1, X2):
    X = X1 + X2

    X_uni = torch.sum(X != 0, dim=(1, 2)).type(torch.float32)
    X_inter = torch.sum(X == 2, dim=(1, 2)).type(torch.float32)

    iou = X_inter / X_uni

    check = (X_inter > 0)

    return iou, check

def draw_plot(data, path):
    x1, y1, metric = data

    step = np.arange(0, 201, 10)
    x1 = x1[step]
    y1 = y1[step]
    plt.rcParams['figure.figsize'] = (11.3, 11.3)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.grid'] = True
    ax = plt.subplot()
    size = 35
    ax.xaxis.set_tick_params(labelsize=size)
    ax.yaxis.set_tick_params(labelsize=size)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_ylabel('Times New Roman')

    plt.xlim(0.05, 0.95)
    plt.ylim(0, 1)
    plt.plot(x1, y1, label="Model", color='deepskyblue', linewidth=3)
    plt.xlabel('Threshold (\u03C4)', fontsize=size)
    legend = plt.legend(loc=3, prop={'size':size})
    legend.get_frame().set_edgecolor('black')

    # plt.show()
    plt.savefig(path + metric + '.png')
    plt.close()


# # pickle
# def save_pickle(dir_name, file_name, data):
#
#     mkdir(dir_name)
#     with open(dir_name + file_name + '.pickle', 'wb') as f:
#         pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# # directory & file
# def mkdir(path):
#     if os.path.exists(path) == False:
#         os.makedirs(path)
