import torch
import torch.nn as nn

class regression_loss(nn.Module):
    def __init__(self):
        super(regression_loss, self).__init__()

    def forward(self, out, gt):

        dist = out - gt
        dist_abs = torch.abs(dist)

        binary_1 = (dist_abs < 1).type(torch.FloatTensor).cuda()
        binary_2 = (dist_abs >= 1).type(torch.FloatTensor).cuda()

        d1 = 0.5 * (dist_abs ** 2)
        d2 = dist_abs - 0.5

        n = torch.mean(torch.sum(d1 * binary_1 + d2 * binary_2, dim=1))

        return n


class Loss_Function(nn.Module):
    def __init__(self):
        super(Loss_Function, self).__init__()
        self.reg_loss = regression_loss()
        self.cls_loss = nn.BCELoss()

    def forward(self, output, gt_cls, gt_reg):

        L1 = self.cls_loss(output['cls'], gt_cls[0])
        L2 = self.reg_loss(output['reg'], gt_reg[0])

        return L1 + L2, L1, L2

