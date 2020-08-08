import os
import pickle

import numpy as np
import random
import torch

global global_seed

global_seed = 123
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)

def _init_fn(worker_id):

    seed = global_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

# convertor
def to_tensor(data):
    return torch.from_numpy(data).cuda()

def to_np(data):
    return data.cpu().numpy()

def to_np2(data):
    return data.detach().cpu().numpy()

def logger(text, LOGGER_FILE):  # write log
    with open(LOGGER_FILE, 'a') as f:
        f.write(text)
        f.close()


# directory & file
def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def rmfile(path):
    if os.path.exists(path):
        os.remove(path)

# pickle
def save_pickle(dir_name, file_name, data):

    '''
    :param file_path: ...
    :param data:
    :return:
    '''
    mkdir(dir_name)
    with open(dir_name + file_name + '.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path + '.pickle', 'rb') as f:
        data = pickle.load(f)

    return data

# create dict
def create_test_dict():

    out = {'cls': {},
           'reg': {},
           'pos': {}}  # detected lines

    # pred
    out['cls'] = torch.FloatTensor([]).cuda()
    out['reg'] = torch.FloatTensor([]).cuda()

    return out

def create_forward_step(num, batch_size):

    step = [i for i in range(0, num, batch_size)]
    if step[len(step) - 1] != num:
        step.append(num)

    return step