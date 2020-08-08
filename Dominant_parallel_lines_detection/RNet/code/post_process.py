import torch

import numpy as np

from libs.modules import *

class Post_Process_CR(object):

    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.forward_model = dict_DB['forward_model']
        self.visualize = dict_DB['visualize']

    def generate_line_pair(self):

        num = self.out_pts[self.rest_idx].shape[0]
        self.rest_num = num

        # reference, target
        idx1 = torch.zeros((num * (num - 1) // 2), dtype=torch.int64).cuda()
        idx2 = torch.zeros((num * (num - 1) // 2), dtype=torch.int64).cuda()

        k = 0
        for i in range(num):
            for j in range(i + 1, num):

                idx1[k] = i
                idx2[k] = j

                k += 1

        self.pairwise = {'idx1': idx1, 'idx2': idx2, 'num': k}

    def run_RNet(self):

        # extract ref & tar line features from DNet
        f_ref = {'fc1': self.out_fc['fc1'][self.pairwise['idx1']],
                 'fc2': self.out_fc['fc2'][self.pairwise['idx1']]}
        f_tar = {'fc1': self.out_fc['fc1'][self.pairwise['idx2']],
                 'fc2': self.out_fc['fc2'][self.pairwise['idx2']]}

        self.out_ranking = self.forward_model.run_comparator(f_ref, f_tar, self.RNet)

    def construct_pairwise_comparison_matrix(self, result):

        self.matrix = torch.zeros((self.rest_num, self.rest_num), dtype=torch.float32)

        for i in range(self.pairwise['num']):
            idx1 = self.pairwise['idx1'][i]
            idx2 = self.pairwise['idx2'][i]

            self.matrix[idx1, idx2] = result['cls'][i, 0]
            self.matrix[idx2, idx1] = result['cls'][i, 1]

    def ranking_and_sorting(self):

        score = torch.sum(self.matrix, dim=1)
        rank_idx = torch.argsort(score, descending=True)
        self.idx_rank_1 = self.rest_idx[int(rank_idx[0])]

        # update
        self.visit[self.idx_rank_1] = 0
        self.rest_idx = (self.visit == 1).nonzero()[:, 0]

        # line selection
        self.dominant_check[self.idx_rank_1] = 1

    def run(self):

        self.rest_idx = (self.visit == 1).nonzero()[:, 0]

        # generate line pair
        self.generate_line_pair()
        self.run_RNet()
        self.construct_pairwise_comparison_matrix(self.out_ranking)
        self.ranking_and_sorting()

        # selected dominant parallel lines
        out_pri = self.out_pts[self.dominant_check == 1]

        return out_pri

    def update_data(self, batch, img, out_pos):
        self.batch = batch

        self.out_pts = out_pos
        self.out_num = self.out_pts.shape[0]

        # feature from detector fc1, fc2
        out = self.forward_model.run_feature_extractor(img=img,
                                                       line_pts=self.out_pts.unsqueeze(0),
                                                       model=self.DNet)

        self.out_fc = {}
        self.out_fc['fc1'] = out['fc1']
        self.out_fc['fc2'] = out['fc2']

        self.visit = torch.ones(self.out_num, dtype=torch.int32)
        self.dominant_check = torch.zeros((self.out_num), dtype=torch.int32)

    def update_model(self, DNet, RNet):

        self.DNet = DNet
        self.RNet = RNet

