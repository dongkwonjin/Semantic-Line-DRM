import os
import cv2
import torch

import numpy as np

from PIL import Image
from libs.modules import *

class Post_Process_NMS(object):

    def __init__(self, cfg=None):
        self.cfg = cfg

        self.height = cfg.height
        self.width = cfg.width

        self.thresd = 0.85

    def compute_edge_density(self, line_pts):

        num = line_pts.shape[0]
        edge_score = torch.zeros(num, dtype=torch.float32).cuda()

        for i in range(num):
            line_mask = np.zeros((self.height, self.width), dtype=np.int32)
            r0, c0, r1, c1 = line_pts[i].cpu().numpy()
            pt_1 = (r0, c0)
            pt_2 = (r1, c1)
            line_mask = cv2.line(line_mask, pt_1, pt_2, (1, 1, 1), 1)
            neighbor_line_mask = cv2.line(line_mask, pt_1, pt_2, (1, 1, 1), 4)

            # HED result
            edge = (np.array(Image.open(self.cfg.dataset_dir + 'edge/' + self.img_name)) > 128)[:, :, 0]

            # compute edge score
            N_all = np.sum(line_mask[:, :])
            N_edge = np.sum(neighbor_line_mask * edge)

            edge_score[i] = N_edge / N_all

        return edge_score

    def run(self):

        # region mask
        mask = divided_region_mask(line_pts=self.line_pts,
                                   size=[self.cfg.width, self.cfg.height])

        # HED
        edge_score = self.compute_edge_density(self.line_pts)

        nms_check = non_maximum_suppression(
            score=edge_score,
            mask=mask,
            thresd=self.thresd)

        nms_out = self.line_pts[nms_check == 0]

        return nms_out

    def update_data(self, batch, line_pts):

        self.batch = batch
        self.img_name = batch['img_name'][0]
        self.line_pts = line_pts
