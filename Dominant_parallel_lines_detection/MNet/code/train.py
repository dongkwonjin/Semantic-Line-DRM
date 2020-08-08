import torch

from libs.save_model import *
from libs.generate_pair import *

class Train_Process_MNet(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

        self.dataloader = dict_DB['trainloader']

        self.DNet = dict_DB['DNet']
        self.RNet = dict_DB['RNet']
        self.MNet = dict_DB['MNet']

        self.optimizer = dict_DB['optimizer']
        self.scheduler = dict_DB['scheduler']
        self.loss_fn = dict_DB['loss_fn']
        self.visualize = dict_DB['visualize']

        self.forward_model = dict_DB['forward_model']

        self.test_process = dict_DB['test_process']
        self.val_result = dict_DB['val_result']

        self.logfile = dict_DB['logfile']
        self.epoch_s = dict_DB['epoch']

    def training(self):

        self.DNet.eval()
        self.MNet.train()
        loss_t = {'sum': 0}

        # train start
        print('train start =====> MNet')
        logger('MNet train start\n', self.logfile)
        for i, batch in enumerate(self.dataloader):

            # load data
            img = batch['img'].cuda()

            # generate line pairs per image
            pairwise = generate_line_pair_MNet(training_line=batch['training_data'],
                                               line_batch_size=self.cfg.batch_size['train_pair'],
                                               mode='training')

            # extract ref & tar line features from detector
            f_ref = self.forward_model.run_feature_extractor(img, pairwise['ref'], self.DNet)
            f_tar = self.forward_model.run_feature_extractor(img, pairwise['tar'], self.DNet)

            # model
            out = self.MNet(f_ref, f_tar)

            # loss
            loss = self.loss_fn(out, pairwise['label'])

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_t['sum'] += loss.item()
            # display
            if i % 100 == 0:
                print('iter %d' % i)
                self.visualize.display_for_train_MNet(batch, pairwise, out, i)
                logger("Loss : %5f\n" % loss.item(), self.logfile)

        # logger
        logger("Average Loss : %5f\n" % (loss_t['sum'] / len(self.dataloader)), self.logfile)
        print("Average Loss : %5f\n" % (loss_t['sum'] / len(self.dataloader)))


        # save model
        self.ckpt = {'epoch': self.epoch,
                     'model': self.MNet,
                     'optimizer': self.optimizer,
                     'val_result': self.val_result}

        save_model(checkpoint=self.ckpt,
                   param='checkpoint_MNet_final',
                   path=self.cfg.weight_dir + '')

    def validation(self):
        AA = self.test_process.run(self.DNet, self.RNet, self.MNet, mode='val')
        logger("Epoch %03d finished... AA : %5f %5f %5f %5f %5f\n"
               % (self.ckpt['epoch'], AA[0], AA[1], AA[2], AA[3], AA[4]), self.logfile)
        self.val_result['AA_10'] = save_model_max(self.ckpt, self.cfg.weight_dir,
                                                  self.val_result['AA_10'], AA[4],
                                                  logger, self.logfile, 'AA_10')

    def run(self):
        for epoch in range(self.epoch_s, self.cfg.epochs):
            self.epoch = epoch
            print('epoch %d' % epoch)
            logger("epoch %d\n" % epoch, self.logfile)
            self.training()
            self.validation()

            self.scheduler.step(self.epoch)

