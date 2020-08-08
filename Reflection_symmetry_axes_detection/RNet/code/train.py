import torch

from libs.save_model import *
from libs.generate_pair import *

class Train_Process_RNet(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

        self.dataloader = dict_DB['trainloader']

        self.DNet = dict_DB['DNet']
        self.RNet = dict_DB['RNet']

        self.optimizer = dict_DB['optimizer']
        self.scheduler = dict_DB['scheduler']
        self.loss_fn = dict_DB['loss_fn']
        self.visualize = dict_DB['visualize']

        self.forward_model = dict_DB['forward_model']

        self.test_process = dict_DB['test_process']
        self.eval_func = dict_DB['eval_func']
        self.val_result = dict_DB['val_result']

        self.logfile = dict_DB['logfile']
        self.epoch_s = dict_DB['epoch']

    def training(self):

        self.DNet.eval()
        self.RNet.train()
        loss_t = {'sum': 0}

        # train start
        print('train start =====> RNet')
        logger('RNet train start\n', self.logfile)
        for i, batch in enumerate(self.dataloader):

            # load data
            img = batch['img'].cuda()
            # generate line pairs per image
            pairwise = generate_line_pair_RNet(training_line=batch['train_data'],
                                               line_batch_size=self.cfg.batch_size['train_pair'],
                                               mode='training')
            if pairwise['num'] == 0:
                print(batch['image_name'])
                continue

            # extract ref & tar line features from detector
            f_ref = self.forward_model.run_feature_extractor(img, pairwise['ref'], self.DNet)
            f_tar = self.forward_model.run_feature_extractor(img, pairwise['tar'], self.DNet)

            # model
            out = self.RNet(f_ref, f_tar)

            # loss
            loss = self.loss_fn(out, pairwise['label'])

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_t['sum'] += loss.item()
            # display
            if i % 10 == 0:
                print('iter %d' % i)
                self.visualize.display_for_train_RNet(batch, pairwise, out, i)
                logger("Loss : %5f\n" % loss.item(), self.logfile)

        # logger
        logger("Average Loss : %5f\n" % (loss_t['sum'] / len(self.dataloader)), self.logfile)
        print("Average Loss : %5f\n" % (loss_t['sum'] / len(self.dataloader)))


        # save model
        self.ckpt = {'epoch': self.epoch,
                     'model': self.RNet,
                     'optimizer': self.optimizer,
                     'val_result': self.val_result}

        save_model(checkpoint=self.ckpt,
                   param='checkpoint_RNet_final',
                   path=self.cfg.weight_dir)

    def validation(self):
        auc_a = self.test_process.run(self.DNet, self.RNet, mode='val')
        logger("AUC_A : %5f\n" % auc_a, self.logfile)
        self.val_result['AUC_A'] = save_model_max(self.ckpt, self.cfg.weight_dir,
                                                  self.val_result['AUC_A'], auc_a,
                                                  logger, self.logfile, 'AUC_A')


    def run(self):
        for epoch in range(self.epoch_s, self.cfg.epochs):
            self.epoch = epoch
            print('epoch %d' % epoch)
            logger("epoch %d\n" % epoch, self.logfile)

            self.training()
            if epoch > 30:
                self.validation()

            self.scheduler.step(self.epoch)

