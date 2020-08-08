import torch


def eval_AUC_PR(out, gt, eval_func):
    miou, match = create_eval_dict()

    num = len(out)

    # eval process
    for i in range(num):
        out[i] = out[i].cuda()
        gt[i] = gt[i].cuda()
        out_num = out[i].shape[0]
        gt_num = gt[i].shape[0]

        if gt_num == 0:
            match['r'][i] = torch.zeros(gt_num, dtype=torch.float32).cuda()
        elif out_num == 0:
            match['p'][i] = torch.zeros(out_num, dtype=torch.float32).cuda()
        else:
            miou['p'][i], miou['r'][i] = eval_func.measure_miou(out=out[i],
                                                                gt=gt[i])
            match['p'][i], match['r'][i] = eval_func.matching(miou=miou,
                                                              idx=i)

    # performance
    auc_p = eval_func.calculate_AUC(miou=match,
                                    metric='precision')

    auc_r = eval_func.calculate_AUC(miou=match,
                                    metric='recall')

    print('---------Performance---------\n'
          'AUC_P %5f / AUC_R %5f' % (auc_p, auc_r))

    return auc_p, auc_r


def eval_AUC_A(out, gt, eval_func):
    miou, match = create_eval_dict()

    num = len(out)

    # eval process
    for i in range(num):
        out[i] = out[i].cuda()
        gt[i] = gt[i].cuda()

        miou['a'][i], _ = eval_func.measure_miou(out=out[i],
                                                 gt=gt[i])

    # performance
    auc_a = eval_func.calculate_AUC(miou=miou,
                                    metric='accuracy')

    print('---------Performance---------\n'
          'AUC_A %5f' % (auc_a))

    return auc_a

def create_eval_dict():
    # a : accuracy / p : precision / r : recall

    miou = {'a': {},
            'p': {},
            'r': {}}

    match = {'p': {},
             'r': {}}

    return miou, match
