import torch
from networks.detector import DNet

from networks.loss import *

def load_DNet_for_test(cfg, dict_DB):
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(cfg.paper_weight_dir + 'checkpoint_DNet_paper')
    else:
        # select ckpt from pretrained_DNet_dir
        checkpoint = torch.load(cfg.weight_dir + 'checkpoint_DNet_final')

    model = DNet(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    dict_DB['DNet'] = model

    return dict_DB


def load_DNet_for_train(cfg, dict_DB):
    model = DNet(cfg)
    model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=cfg.milestones,
                                                     gamma=cfg.gamma)

    if cfg.resume == False:
        checkpoint = torch.load(cfg.weight_dir + 'checkpoint_DNet_final')
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=cfg.milestones,
                                                         gamma=cfg.gamma,
                                                         last_epoch=checkpoint['epoch'])
        dict_DB['epoch'] = checkpoint['epoch']
        dict_DB['val_result'] = checkpoint['val_result']

    loss_fn = Loss_Function()

    dict_DB['DNet'] = model
    dict_DB['optimizer'] = optimizer
    dict_DB['scheduler'] = scheduler
    dict_DB['loss_fn'] = loss_fn

    return dict_DB