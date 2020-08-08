import torch

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

        # extract ref & tar line features (line pooling) from detector
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

        self.select_semantic_line(self.idx_rank_1)


    def run_MNet(self, idx_rank_1):

        c = self.out_fc['fc1'][idx_rank_1].shape[0]

        f_ref2 = {'fc1': self.out_fc['fc1'][idx_rank_1].unsqueeze(0).expand(torch.sum(self.visit), c)}
        f_tar2 = {'fc1': self.out_fc['fc1'][self.rest_idx]}

        out = self.forward_model.run_comparator(f_ref2, f_tar2, self.MNet)
        self.out_matching = torch.argmax(out['cls'], dim=1)

    def line_removal(self, result):

        self.visit[self.rest_idx[result == 0]] = 0
        self.rest_idx = (self.visit == 1).nonzero()[:, 0]

    def select_semantic_line(self, idx):
        # primary & multiple line check

        if self.iter == 0:
            self.pri_check[idx] = 1

        self.mul_check[idx] = self.iter + 1


    def run(self):

        # iteration
        self.rest_idx = (self.visit == 1).nonzero()[:, 0]
        for self.iter in range(self.out_num):

            if torch.sum(self.visit) == 0:  # all check
                break
            if torch.sum(self.visit) == 1:  # only one line
                self.select_semantic_line((self.visit == 1).nonzero()[0])
                break

            # generate line pair
            self.generate_line_pair()
            self.run_RNet()
            self.construct_pairwise_comparison_matrix(self.out_ranking)
            self.ranking_and_sorting()
            self.run_MNet(self.idx_rank_1)
            self.line_removal(self.out_matching)


        # selected semantic line
        out_pri = self.out_pts[self.pri_check == 1]
        out_mul = self.out_pts[self.mul_check != 0]


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
        self.pri_check = torch.zeros(self.out_num, dtype=torch.int32)
        self.mul_check = torch.zeros(self.out_num, dtype=torch.int32)

    def update_model(self, DNet, RNet, MNet):

        self.DNet = DNet
        self.RNet = RNet
        self.MNet = MNet
