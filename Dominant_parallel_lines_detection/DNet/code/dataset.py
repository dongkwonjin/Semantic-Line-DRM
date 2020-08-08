import os
import torch
import random
import pickle

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data.dataset import Dataset

from libs.utils import *
from libs.neurvps_func import *

class Train_Dataset_DNet(Dataset):
    def __init__(self, cfg):

        self.cfg = cfg

        # image size
        self.height = cfg.height
        self.width = cfg.width


        # load datalist
        with open(cfg.pickle_dir + 'DNet/train_candidates.pickle', 'rb') as f:
            self.datalist = pickle.load(f)

        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), 2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

        self.pos_num = int(cfg.batch_size['train_line'] * 0.4)
        self.neg_num = int(cfg.batch_size['train_line'] * 0.6)

    def get_image(self, flip, idx):
        img_name = self.cfg.dataset_dir + self.datalist[idx]['img_path']
        img = Image.open(img_name).convert('RGB')
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        width, height = img.size
        return img, torch.FloatTensor([height, width]), self.datalist[idx]['img_path']

    def get_data(self, flip, idx):

        # load pickle
        data = self.datalist[idx]

        pos_data = np.concatenate((data['pos_line']['endpts'], data['pos_line']['cls'], data['pos_line']['offset']), axis=1)
        neg_data = np.concatenate((data['neg_line']['endpts'], data['neg_line']['cls'], data['neg_line']['offset']), axis=1)

        # random sampling
        pos_idx = torch.randperm(pos_data.shape[0])[:self.pos_num]
        neg_idx = torch.randperm(neg_data.shape[0])[:self.neg_num]

        pos_data = pos_data[pos_idx]
        neg_data = neg_data[neg_idx]


        if flip == 1:
            pos_data[:, 0] = self.width - 1 - pos_data[:, 0]
            pos_data[:, 2] = self.width - 1 - pos_data[:, 2]
            pos_data[:, 6] = -1 * pos_data[:, 6]
            pos_data[:, 8] = -1 * pos_data[:, 8]

            neg_data[:, 0] = self.width - 1 - neg_data[:, 0]
            neg_data[:, 2] = self.width - 1 - neg_data[:, 2]
            neg_data[:, 6] = -1 * neg_data[:, 6]
            neg_data[:, 8] = -1 * neg_data[:, 8]

        return pos_data, neg_data

    def normalize_point_coord(self, data):
        data[:, 6] = data[:, 6] / self.cfg.width
        data[:, 7] = data[:, 7] / self.cfg.height
        data[:, 8] = data[:, 8] / self.cfg.width
        data[:, 9] = data[:, 9] / self.cfg.height
        return data



    def __getitem__(self, idx):
        # get pre-processed images

        flip = random.randint(0, 1)

        img, img_size, img_name = self.get_image(flip, idx)
        img = self.transform(img)

        # get candidate lines
        pos_data, neg_data = self.get_data(flip, idx)
        pos_data = self.normalize_point_coord(pos_data)

        train_data = np.concatenate((pos_data, neg_data), axis=0)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'img_size': img_size,
                'img_name': img_name,
                'flip': flip,
                'train_data': train_data}

    def __len__(self):
        return len(self.datalist)



class AVA_Test_Dataset(Dataset):

    def __init__(self, cfg):
        # cfg
        self.cfg = cfg

        # image size
        self.height = cfg.height
        self.width = cfg.width

        # load datalist
        with open(self.cfg.dataset_dir + 'neurvps_test.pickle', 'rb') as f:
            self.datalist = pickle.load(f)

        # image transform
        self.transform = transforms.Compose([transforms.Resize((self.height, self.width), 2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def get_image(self, flip, idx):

        img_path = self.cfg.dataset_dir + self.datalist['img_path'][idx]

        img = Image.open(img_path).convert('RGB')
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        width, height = img.size

        return img, torch.FloatTensor([height, width]), self.datalist['img_path'][idx]

    def get_vp(self, img_size, idx):

        label = (self.cfg.dataset_dir + self.datalist['img_path'][idx]).replace('.jpg', '.txt')
        axy, bxy = np.genfromtxt(label, skip_header=1)

        a0, a1 = np.array(axy[:2]), np.array(axy[2:])
        b0, b1 = np.array(bxy[:2]), np.array(bxy[2:])
        xy = intersect(a0, a1, b0, b1) - 0.5

        xy[0] *= self.width / img_size[1]
        xy[1] *= self.height / img_size[0]

        vpts = np.array([[xy[0] / (self.width / 2) - 1, 1 - xy[1] / (self.height / 2), 1]])
        vpts[0] /= LA.norm(vpts[0])
        vpts = np.float32(vpts)
        return self.datalist['gtline'][idx], vpts, xy


    def __getitem__(self, idx):

        # flip
        flip = 0

        # get pre-processed images
        img, img_size, img_name = self.get_image(flip, idx)
        img = self.transform(img)

        gt, vp, vp_xy = self.get_vp(img_size, idx)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'img_size': img_size,
                'img_name': img_name,
                'flip': flip,
                'gt': gt,
                'vp': vp,
                'vp_xy': vp_xy}

    def __len__(self):
        return len(self.datalist['img_path'])
