import numpy as np

from evaluation.eval_process import *
from libs.utils import *

class Test_Process_DR(object):
    def __init__(self, cfg, dict_DB):

        self.cfg = cfg
        self.dataloader = dict_DB['testloader']

        self.DNet = dict_DB['DNet']
        self.RNet = dict_DB['RNet']

        self.forward_model = dict_DB['forward_model']

        self.post_process = dict_DB['CR_process']

        self.eval_func = dict_DB['eval_func']

        self.batch_size = self.cfg.batch_size['test_line']
        self.size = to_tensor(np.float32(cfg.size))
        self.candidates = load_pickle(self.cfg.pickle_dir + 'detector/detector_test_candidates')
        self.candidates = to_tensor(self.candidates).unsqueeze(0)
        self.cand_num = self.candidates.shape[1]
        self.step = create_forward_step(self.candidates.shape[1],
                                        cfg.batch_size['test_line'])

        self.visualize = dict_DB['visualize']

    def run(self, DNet, RNet, mode='test'):
        result = {'out': {'pri': [], 'mul': []},
                  'gt': {'pri': [], 'mul': []}}

        with torch.no_grad():
            DNet.eval()
            RNet.eval()

            self.post_process.update_model(DNet, RNet)

            for i, self.batch in enumerate(self.dataloader):  # load batch data

                self.img_name = self.batch['img_name'][0]
                self.img = self.batch['img'].cuda()
                pri_gt = self.batch['pri_gt'][0][:, :4]

                # semantic line detection
                out = self.forward_model.run_detector(img=self.img,
                                                      line_pts=self.candidates,
                                                      step=self.step,
                                                      model=DNet)
                # reg result
                out['pts'] = self.candidates[0] + out['reg'] * self.size
                # cls result
                pos_check = torch.argmax(out['cls'], dim=1)
                out['pos'] = out['pts'][pos_check == 1]
                # primary line
                sorted = torch.argsort(out['cls'][:, 1], descending=True)
                out['pri'] = out['pts'][sorted[0], :].unsqueeze(0)
                if torch.sum(pos_check == 1) == 0:
                    out['pos'] = out['pri']

                # post process
                if out['pos'].shape[0] > 1:
                    self.post_process.update_data(self.batch, self.img, out['pos'])
                    out['pri'] = self.post_process.run()

                # visualize
                self.visualize.display_for_test(batch=self.batch, out=out)


                # record output data
                result['out']['pri'].append(out['pri'])
                result['gt']['pri'].append(pri_gt)

                print('image %d ---> %s done!' % (i, self.img_name))


        # save pickle
        save_pickle(dir_name=self.cfg.output_dir + 'test/pickle/',
                    file_name='result',
                    data=result)

        return self.evaluation()

    def evaluation(self):
        # evaluation
        data = load_pickle(self.cfg.output_dir + 'test/pickle/result')
        auc_a = eval_AUC_A(out=data['out']['pri'],
                           gt=data['gt']['pri'],
                           eval_func=self.eval_func)

        return auc_a