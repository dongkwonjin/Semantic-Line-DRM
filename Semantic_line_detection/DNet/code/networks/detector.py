import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import math

import numpy as np

from libs.utils import *

class FeatureExtraction(nn.Module):
    def __init__(self, feature_extraction_cnn='vgg16'):
        super(FeatureExtraction, self).__init__()

        if feature_extraction_cnn == 'vgg16':
            model = models.vgg16(pretrained=True)

            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
                                   'pool1', 'conv2_1', 'relu2_1', 'conv2_2',
                                   'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                                   'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
                                   'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
                                   'relu4_2', 'conv4_3', 'relu4_3']

            last_layer = 'relu4_3'
            last_layer_idx = vgg_feature_layers.index(last_layer)

            self.model1 = nn.Sequential(*list(model.features.children())[:last_layer_idx+1])
            self.model2 = nn.Sequential(*list(model.features.children())[last_layer_idx+1:-1])

    def forward(self, img):
        feat1 = self.model1(img)
        feat2 = self.model2(feat1)

        return feat1, feat2

class Fully_connected_layer(nn.Module):
    def __init__(self):
        super(Fully_connected_layer, self).__init__()

        self.linear_1 = nn.Linear(2048, 1024)
        self.linear_2 = nn.Linear(1024, 1024)


    def forward(self, x):
        x = x.view(x.size(0), -1)

        fc1 = self.linear_1(x)
        fc1 = F.relu(fc1)
        fc2 = self.linear_2(fc1)
        fc2 = F.relu(fc2)

        return fc1, fc2

class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()

        self.linear = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.linear(x)
        x = F.softmax(x, dim=1)

        return x

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()

        self.linear = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.linear(x)

        return x


class DNet(nn.Module):
    def __init__(self, cfg, visualize=None):
        super(DNet, self).__init__()
        size = torch.FloatTensor([cfg.width, cfg.height, cfg.width, cfg.height]).cuda()

        self.feature_extraction = FeatureExtraction()

        self.fully_connected = Fully_connected_layer()
        self.regression = Regression()
        self.classification = Classification()

        self.region_pooling = Region_Pooling_Module(cfg=cfg, size=size)


    def forward(self, img, line_pts, with_comparator=False, feat1=None, feat2=None):
        # Feature extraction
        if feat1 is None:
            feat1, feat2 = self.feature_extraction(img)  # 512,50,50 // 512,25,25

        rp1 = self.region_pooling(feat1, line_pts, 8)
        rp2 = self.region_pooling(feat2, line_pts, 16)

        rp_concat = torch.cat((rp1, rp2), dim=1)
        fc_out1, fc_out2 = self.fully_connected(rp_concat)  #

        if with_comparator is False:

            # classification & regression
            reg_out = self.regression(fc_out2)
            cls_out = self.classification(fc_out2)

            return {'reg': reg_out, 'cls': cls_out}
        else:

            return {'fc1': fc_out1, 'fc2': fc_out2}



class Region_Pooling_Module(nn.Module):
    def __init__(self, cfg, size):
        super(Region_Pooling_Module, self).__init__()
        self.size = size

        self.separate_region = Separate_Region(cfg=cfg, size=size)
        self.gaussian_filter = Gaussian_Filter()

        self.MA = Mirror_Attention().cuda()

        # Grid generation
        width, height = int(size[0]), int(size[1])

        X, Y = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height)) # after x before
        self.X = torch.tensor(X, dtype=torch.float, requires_grad=False).unsqueeze(0).cuda()
        self.Y = torch.tensor(Y, dtype=torch.float, requires_grad=False).unsqueeze(0).cuda()

    def forward(self, feat_map, line_pts, ratio):
        b = line_pts.shape[1]

        # scale line pts
        norm_pts = line_pts[:, :, :4] / (self.size - 1) * (self.size / ratio - 1)
        norm_pts2 = (norm_pts / (self.size / ratio - 1) - 0.5) * 2  # [-1, 1]

        # expand batch size of feature map
        _, c, h, w = feat_map.shape
        feat_map = feat_map.expand(b, c, h, w)

        # get line mask and two region's mask divided by a candidate line
        region_mask, line_mask = self.separate_region(norm_pts[0], self.size // ratio)

        # blur the line mask using gaussian filter
        blurred_line_mask = self.gaussian_filter.blurring(line_mask)
        region_mask = region_mask * blurred_line_mask

        region1 = (region_mask[:, 0, :, :].unsqueeze(1) != 0).type(torch.float)
        region2 = (region_mask[:, 1, :, :].unsqueeze(1) != 0).type(torch.float)

        n1 = torch.sum(region1, dim=(1, 2, 3), keepdim=True) + 1e-9
        n2 = torch.sum(region2, dim=(1, 2, 3), keepdim=True) + 1e-9

        # gaussian filtering the feature map
        g_map = blurred_line_mask * feat_map

        # get points symmetrical to the line using line equation
        l_eq, check = line_equation(norm_pts2[0])

        X2, Y2 = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))  # after x before
        grid_X0 = torch.tensor(X2, dtype=torch.float, requires_grad=False).unsqueeze(0).cuda()
        grid_Y0 = torch.tensor(Y2, dtype=torch.float, requires_grad=False).unsqueeze(0).cuda()
        grid_X0 = grid_X0.expand(norm_pts2.shape[1], h, w)
        grid_Y0 = grid_Y0.expand(norm_pts2.shape[1], h, w)

        l_eq = l_eq.view(norm_pts2.shape[1], 1, 1, 3)
        grid_X1, grid_Y1 = point_flip(l_eq, check, norm_pts2, grid_X0, grid_Y0)

        grid_XY = torch.cat((grid_X1.unsqueeze(3), grid_Y1.unsqueeze(3)), dim=3)

        # line_symmetric_attention module
        att_map = self.MA(g_map, grid_XY)
        # I + A (residual)
        att_g_map = g_map + g_map * att_map.expand_as(g_map)

        feat_r1 = att_g_map * region1
        feat_r2 = att_g_map * region2

        # average pooling
        f_rp1 = torch.sum(feat_r1, dim=(2, 3), keepdim=True) / n1
        f_rp2 = torch.sum(feat_r2, dim=(2, 3), keepdim=True) / n2

        f_rp = torch.cat((f_rp1, f_rp2), dim=1)

        return f_rp

class Mirror_Attention(nn.Module):
    def __init__(self):
        super(Mirror_Attention, self).__init__()

        self.process = nn.Sequential(
            # process
            nn.Conv2d(1024, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid())

    def forward(self, x, grid):

        # fold feature map and convolution
        x_fold = F.grid_sample(x, grid, align_corners=True)
        x_concat = torch.cat((x, x_fold), dim=1)
        out = self.process(x_concat)

        return out

class Separate_Region(object):

    def __init__(self, cfg, size):
        super(Separate_Region, self).__init__()
        self.cfg = cfg
        self.size = size

        lap_filter = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.double).cuda()
        self.lap_filter = lap_filter.view(1, 1, 3, 3)

        width, height = int(size[0]), int(size[1])

        X, Y = np.meshgrid(np.linspace(0, height - 1, height), np.linspace(0, width - 1, width))  # after x before
        self.X = torch.tensor(X, dtype=torch.float, requires_grad=False).unsqueeze(0).cuda()
        self.Y = torch.tensor(Y, dtype=torch.float, requires_grad=False).unsqueeze(0).cuda()

    def __call__(self, line_pts, size):

        line_num, _ = line_pts.shape

        width, height = int(size[0]), int(size[1])

        X = self.X[:, :width, :height]
        Y = self.Y[:, :width, :height]

        check = ((line_pts[:, 1] - line_pts[:, 3]) == 0).type(torch.float)

        mask1 = torch.zeros((line_num, width, height), dtype=torch.float32).cuda()
        mask2 = torch.zeros((line_num, width, height), dtype=torch.float32).cuda()

        mask1[check == 1, :, :] = (X < line_pts[:, 1].view(line_num, 1, 1)).type(torch.float)[check == 1, :, :]
        mask2[check == 1, :, :] = (X > line_pts[:, 1].view(line_num, 1, 1)).type(torch.float)[check == 1, :, :]

        a = (line_pts[:, 0] - line_pts[:, 2]) / (line_pts[:, 1] - line_pts[:, 3])
        b = -1 * a * line_pts[:, 1] + line_pts[:, 0]

        a = a.view(line_num, 1, 1)
        b = b.view(line_num, 1, 1)

        mask1[check == 0, :, :] = (Y < a * X + b).type(torch.float32)[check == 0, :, :]
        mask2[check == 0, :, :] = (Y > a * X + b).type(torch.float32)[check == 0, :, :]

        mask = torch.cat((mask1.unsqueeze(1), mask2.unsqueeze(1)), dim=1)

        edge_mask = self.edge_detector(mask)

        return mask.permute(0, 1, 3, 2), edge_mask.permute(0, 1, 3, 2)

    def edge_detector(self, mask):

        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)

        mask = mask.type(torch.double)

        edge = F.conv2d(F.pad(mask[:, 0, :, :].unsqueeze(1), [1, 1, 1, 1], mode='replicate'), self.lap_filter)
        edge = edge.type(torch.float)
        edge[edge != 0] = 1
        return edge

class Gaussian_Filter(object):
    def __init__(self):
        self.gaussian_kernel = self.get_gaussian_kernel()

    def get_gaussian_kernel(self, kernel_size=5, sigma=5, channels=1):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        self.pad = int(kernel_size / 2)

        return gaussian_filter.cuda()

    def blurring(self, mask):
        mask_blur = self.gaussian_kernel(F.pad(mask, (self.pad, self.pad, self.pad, self.pad), mode='replicate'))

        b, c, h, w,  = mask_blur.shape
        max_val = torch.max(mask_blur.contiguous().view(b, c, h*w), dim=2)[0].view(b, 1, 1, 1) + 1e-9
        mask_blur = mask_blur / max_val
        return mask_blur

def line_equation(data):
    '''
    :param data: [N, 4] numpy array  x1, y1, x2, y2 (W, H, W, H)
    :return:
    '''
    line_eq = torch.zeros((data.shape[0], 3)).cuda()
    line_eq[:, 0] = (data[:, 1] - data[:, 3]) / (data[:, 0] - data[:, 2])
    line_eq[:, 1] = -1
    line_eq[:, 2] = -1 * line_eq[:, 0] * data[:, 0] + data[:, 1]
    check = ((data[:, 0] - data[:, 2]) == 0)
    return line_eq, check

def point_flip(l_eq, check, pts, x0, y0):
    a = l_eq[:, :, :, 0]
    b = l_eq[:, :, :, 1]
    c = l_eq[:, :, :, 2]

    x1 = x0 - 2 * a * (a * x0 + b * y0 + c)\
         / (a * a + b * b)
    y1 = y0 - 2 * b * (a * x0 + b * y0 + c)\
         / (a * a + b * b)

    # inf check
    d = x0[check == 1] - pts[0, check == 1, 0].view(-1, 1, 1)
    x1[check == 1] = x0[check == 1] - d * 2
    y1[check == 1] = y0[check == 1]

    return x1, y1
