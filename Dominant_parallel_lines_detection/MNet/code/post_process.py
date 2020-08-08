import torch

import numpy as np

from libs.modules import *

class Post_Process_CRM(object):

    def __init__(self, dict_DB):

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

    def run_MNet(self):

        c = self.out_fc['fc1'][self.idx_rank_1].shape[0]

        f_ref2 = {'fc1': self.out_fc['fc1'][self.idx_rank_1].unsqueeze(0).expand(torch.sum(self.visit), c)}
        f_tar2 = {'fc1': self.out_fc['fc1'][self.rest_idx]}

        out = self.forward_model.run_comparator(f_ref2, f_tar2, self.MNet)

        idx_matching_1 = torch.argsort(out['cls'][:, 1], descending=True)
        idx_matching_1 = self.rest_idx[int(idx_matching_1[0])]

        # line selection
        self.dominant_check[idx_matching_1] = 2

    def run(self):

        self.rest_idx = (self.visit == 1).nonzero()[:, 0]

        # generate line pair
        self.generate_line_pair()
        self.run_RNet()
        self.construct_pairwise_comparison_matrix(self.out_ranking)
        self.ranking_and_sorting()
        self.run_MNet()


        # selected dominant parallel lines
        out_pri = self.out_pts[self.dominant_check == 1]
        out_mul = self.out_pts[self.dominant_check != 0]

        return out_pri, out_mul

    def update_data(self, batch, img, out_pos):
        self.batch = batch

        self.out_pos = out_pos
        self.out_pts = self.out_pos
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

    def update_model(self, DNet, RNet, MNet):

        self.DNet = DNet
        self.RNet = RNet
        self.MNet = MNet


class Post_Process_CRM_removal(object):

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

    def run_MNet(self):

        c = self.out_fc['fc1'][self.idx_rank_1].shape[0]

        f_ref2 = {'fc1': self.out_fc['fc1'][self.idx_rank_1].unsqueeze(0).expand(torch.sum(self.visit), c)}
        f_tar2 = {'fc1': self.out_fc['fc1'][self.rest_idx]}

        out = self.forward_model.run_comparator(f_ref2, f_tar2, self.MNet)

        pos_check = torch.argmax(out['cls'], dim=1)
        pos_cls = out['cls'][pos_check == 1, 1]

        # remove negative lines
        neg_idx = self.rest_idx[(pos_check == 0).nonzero()[:, 0]]
        self.visit[neg_idx] = 0
        self.rest_idx = (self.visit == 1).nonzero()[:, 0]

        return pos_cls

    def line_removal(self, top_m):
        # edge density
        res_pts = self.out_pts[self.rest_idx]

        # region mask
        mask = divided_region_mask(line_pts=res_pts,
                                   size=[self.cfg.width, self.cfg.height])
        check = suppression(top_m, mask, 0.85).type(torch.float32)


        # update
        self.visit[self.rest_idx[top_m]] = 0  # top matching
        self.visit[self.rest_idx[check == 1]] = 0  # suppressed
        self.rest_idx = (self.visit == 1).nonzero()[:, 0]

        return check


    def run(self):

        self.rest_idx = (self.visit == 1).nonzero()[:, 0]

        # generate line pair
        self.generate_line_pair()
        self.run_RNet()
        self.construct_pairwise_comparison_matrix(self.out_ranking)
        self.ranking_and_sorting()
        out_cls = self.run_MNet()

        num = np.minimum(self.top_k, self.out_num)
        for iter in range(num - 1):
            if out_cls.shape[0] == 0: # all suppressed
                break
            sorted = torch.argsort(out_cls, descending=True)
            top_m = self.rest_idx[int(sorted[0])]
            self.dominant_check[top_m] = 2

            top_check = torch.zeros((sorted.shape[0]), dtype=torch.int32).cuda()
            top_check[sorted[0]] = 1


            remove_check = self.line_removal(sorted[0])

            out_cls = out_cls[(remove_check + top_check) == 0]

        # selected dominant parallel lines
        out_pri = self.out_pts[self.dominant_check == 1]
        out_mul = self.out_pts[self.dominant_check != 0]

        return out_pri, out_mul

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

    def update_model(self, DNet, RNet, MNet, top_k):

        self.top_k = top_k
        self.DNet = DNet
        self.RNet = RNet
        self.MNet = MNet
