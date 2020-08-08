import torch

from libs.save_model import *
from libs.utils import *

class Train_Process_DNet(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

        self.dataloader = dict_DB['trainloader']
        self.DNet = dict_DB['DNet']
        self.optimizer = dict_DB['optimizer']
        self.scheduler = dict_DB['scheduler']
        self.loss_fn = dict_DB['loss_fn']
        self.visualize = dict_DB['visualize']

        self.test_process = dict_DB['test_process']
        self.eval_func = dict_DB['eval_func']
        self.val_result = dict_DB['val_result']

        self.logfile = dict_DB['logfile']
        self.epoch_s = dict_DB['epoch']

    def training(self):

        self.DNet.train()
        loss_t = {'sum': 0, 'cls': 0, 'reg': 0}

        # train start
        print('train start =====> DNet')
        logger('DNet train start\n', self.logfile)
        for i, batch in enumerate(self.dataloader):

            # shuffle idx with pos:neg = 4:6
            idx = torch.randperm(self.cfg.batch_size['train_line'])
            batch['train_data'] = batch['train_data'].cuda()
            batch['train_data'][0, :] = batch['train_data'][0, idx]
            # load data
            img = batch['img'].cuda()
            candidates = batch['train_data'][:, :, :4]
            gt_cls = batch['train_data'][:, :, 4:6]
            gt_reg = batch['train_data'][:, :, 6:]
            # model
            out = self.DNet(img=img,
                            line_pts=candidates)

            # loss
            loss, loss_cls, loss_reg = self.loss_fn(output=out,
                                                    gt_cls=gt_cls,
                                                    gt_reg=gt_reg)

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_t['sum'] += loss.item()
            loss_t['cls'] += loss_cls.item()
            loss_t['reg'] += loss_reg.item()
            # display
            if i % 100 == 0:
                print('iter %d' % i)
                self.visualize.display_for_train_detector(batch, out, i)
                logger("Loss : %5f, "
                       "Loss_cls : %5f, "
                       "Loss_reg : %5f\n"
                       % (loss.item(), loss_cls.item(), loss_reg.item()), self.logfile)

        # logger
        logger("Average Loss : %5f %5f %5f\n"
               % (loss_t['sum'] / len(self.dataloader),
                  loss_t['cls'] / len(self.dataloader),
                  loss_t['reg'] / len(self.dataloader)), self.logfile)
        print("Average Loss : %5f %5f %5f\n"
              % (loss_t['sum'] / len(self.dataloader),
                 loss_t['cls'] / len(self.dataloader),
                 loss_t['reg'] / len(self.dataloader)))


        # save model
        self.ckpt = {'epoch': self.epoch,
                     'model': self.DNet,
                     'optimizer': self.optimizer,
                     'val_result': self.val_result}
        save_model(checkpoint=self.ckpt,
                   param='checkpoint_DNet_final',
                   path=self.cfg.weight_dir)

    def validation(self):
        auc_a, auc_p, auc_r = self.test_process.run(self.DNet, mode='val')
        logger("Epoch %03d finished... AUC_A %5f / AUC_P %5f / AUC_R %5f\n" % (self.ckpt['epoch'], auc_a, auc_p, auc_r), self.logfile)
        self.val_result['AUC_R_upper_P_0.80'] = save_model_max_upper(self.ckpt, self.cfg.weight_dir,
                                                                     self.val_result['AUC_R_upper_P_0.80'], auc_r, auc_p, 0.80,
                                                                     logger, self.logfile, 'AUC_R_upper_P_0.80')
        self.val_result['AUC_R_upper_P_0.78'] = save_model_max_upper(self.ckpt, self.cfg.weight_dir,
                                                                     self.val_result['AUC_R_upper_P_0.78'], auc_r, auc_p, 0.78,
                                                                     logger, self.logfile, 'AUC_R_upper_P_0.78')

    def run(self):
        for epoch in range(self.epoch_s, self.cfg.epochs):
            self.epoch = epoch
            print('epoch %d' % epoch)
            logger("epoch %d\n" % epoch, self.logfile)
            self.training()
            if epoch > 40:
                self.validation()

            self.scheduler.step(self.epoch)

