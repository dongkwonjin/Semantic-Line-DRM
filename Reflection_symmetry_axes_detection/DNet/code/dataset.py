import os
import torch
import random
import pickle

import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image
from torch.utils.data.dataset import Dataset

from libs.utils import *
from libs.modules import *
from libs.line_generator import *

class Train_Dataset_DNet(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_gt = load_pickle(cfg.train_dataset_dir + 'data/train_s')
        self.datalist = self.train_gt['img_path']

        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), 2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

        self.pos_num = int(cfg.batch_size['train_line'] * cfg.ratio_pos)
        self.neg_num = int(cfg.batch_size['train_line'] * (1 - cfg.ratio_pos))

        self.height = cfg.height
        self.width = cfg.width
        self.size = np.float32(cfg.size)[0:2]

        self.line_generator = Training_Line_Generator(cfg)
        _ = self.line_generator.candidate_line()

    def get_image(self, idx):
        img_name = os.path.join(self.cfg.train_img_dir, self.datalist[idx])
        img = Image.open(img_name).convert('RGB')
        width, height = img.size
        return img, torch.FloatTensor([height, width]), self.datalist[idx]

    def get_flipped_data(self, img, label, flip):
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label[:, 0] = self.width - 1 - label[:, 0]
            label[:, 2] = self.width - 1 - label[:, 2]

        return img, label

    def process_data(self, pos_data, neg_data):

        # random sampling
        pos_data = np.concatenate((pos_data['pos']['endpts'],
                                   pos_data['pos']['cls'],
                                   pos_data['pos']['offset']), axis=1)
        neg_data = np.concatenate((neg_data['neg']['endpts'],
                                   neg_data['neg']['cls'],
                                   neg_data['neg']['offset']), axis=1)

        pos_idx = torch.randperm(pos_data.shape[0])[:self.pos_num]
        neg_idx = torch.randperm(neg_data.shape[0])[:self.neg_num]

        pos_data = pos_data[pos_idx]
        neg_data = neg_data[neg_idx]

        return pos_data, neg_data

    def normalize_point_coord(self, data):
        data[:, 6] = data[:, 6] / self.cfg.width
        data[:, 7] = data[:, 7] / self.cfg.height
        data[:, 8] = data[:, 8] / self.cfg.width
        data[:, 9] = data[:, 9] / self.cfg.height
        return data

    def __getitem__(self, idx):
        flip = random.randint(0, 1)

        # get pre-processed images
        img, img_size, img_name = self.get_image(idx)
        label = self.train_gt['label'][idx]
        img, label = self.get_flipped_data(img, label, flip)
        img = self.transform(img)

        # Augmentation by rotation
        theta_aff, theta, scale = random_rotation(self.height)
        aff_grid = F.affine_grid(theta_aff, torch.Size((1, 2, self.height, self.width)), align_corners=True)
        rotated_img = F.grid_sample(img.unsqueeze(0), aff_grid, align_corners=True)[0]

        rotated_gt = np.zeros((4), dtype=np.float32)
        rotated_gt[:2] = (np.matmul(np.linalg.inv(theta_aff[0, :, :2].cpu().numpy()), label[0, :2] / (self.size - 1) - 0.5) + 0.5) * (self.size - 1)
        rotated_gt[2:] = (np.matmul(np.linalg.inv(theta_aff[0, :, :2].cpu().numpy()), label[0, 2:] / (self.size - 1) - 0.5) + 0.5) * (self.size - 1)
        gtline = np.expand_dims(find_endpoints(rotated_gt, self.size - 1), 0)


        # get candidate lines
        self.line_generator.update(gtline=gtline)
        pos_data = self.line_generator.positive_line()
        neg_data = self.line_generator.negative_line()
        pos_data, neg_data = self.process_data(pos_data, neg_data)
        pos_data = self.normalize_point_coord(pos_data)

        train_data = np.concatenate((pos_data, neg_data), axis=0)

        return {'img_rgb': rotated_img,
                'img': self.normalize(rotated_img),
                'img_size': img_size,
                'img_name': img_name,
                'flip': flip,
                'train_data': train_data}

    def __len__(self):
        return len(self.datalist)



class ICCV_Test_Dataset(Dataset):
    def __init__(self, cfg):

        self.cfg = cfg

        # load datalist
        self.datalist = load_pickle(cfg.test_dataset_dir + 'data/test_s')
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), 2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

        self.height = cfg.height
        self.width = cfg.width

    def get_image(self, idx):
        img_name = os.path.join(self.cfg.test_img_dir,
                                self.datalist['img_path'][idx])
        img = Image.open(img_name).convert('RGB')

        width, height = img.size
        return img, torch.FloatTensor([height, width]), self.datalist['img_path'][idx]

    def get_gtlines(self, idx):

        gt = self.datalist['label'][idx]
        return gt


    def __getitem__(self, idx):

        # get pre-processed images
        img, img_size, img_name = self.get_image(idx)
        img = self.transform(img)

        gt = self.get_gtlines(idx)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'img_size': img_size,
                'img_name': img_name,
                'gt': gt}

    def __len__(self):
        return len(self.datalist['img_path'])

class NYU_Test_Dataset(Dataset):
    def __init__(self, cfg):

        self.cfg = cfg

        # load datalist
        self.datalist = load_pickle(cfg.test_dataset_dir + 'data/test_s')
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), 2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

        self.height = cfg.height
        self.width = cfg.width

    def get_image(self, idx):
        img_name = os.path.join(self.cfg.test_img_dir,
                                self.datalist['img_path'][idx])
        img = Image.open(img_name).convert('RGB')

        width, height = img.size
        return img, torch.FloatTensor([height, width]), self.datalist['img_path'][idx]

    def get_gtlines(self, idx):

        gt = self.datalist['label'][idx]
        return gt


    def __getitem__(self, idx):

        # get pre-processed images
        img, img_size, img_name = self.get_image(idx)
        img = self.transform(img)

        gt = self.get_gtlines(idx)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'img_size': img_size,
                'img_name': img_name,
                'gt': gt}

    def __len__(self):
        return len(self.datalist['img_path'])

class SYM_Hard_Test_Dataset(Dataset):
    def __init__(self, cfg):

        self.cfg = cfg

        # load datalist
        self.datalist = load_pickle(cfg.test_dataset_dir + 'data/test_hard')
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), 2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

        self.height = cfg.height
        self.width = cfg.width

    def get_image(self, idx):
        img_name = os.path.join(self.cfg.test_img_dir,
                                self.datalist['img_path'][idx])
        img = Image.open(img_name).convert('RGB')

        width, height = img.size
        return img, torch.FloatTensor([height, width]), self.datalist['img_path'][idx]

    def get_gtlines(self, idx):

        gt = self.datalist['label'][idx]
        return gt


    def __getitem__(self, idx):

        # get pre-processed images
        img, img_size, img_name = self.get_image(idx)
        img = self.transform(img)

        gt = self.get_gtlines(idx)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'img_size': img_size,
                'img_name': img_name,
                'gt': gt}

    def __len__(self):
        return len(self.datalist['img_path'])


def random_rotation(size):
    # theta < 0 -> clock-wise
    theta = np.float32(random.randrange(-90, 90))
    alpha = theta / 180 * np.pi

    if theta >= 0:
        alpha_2 = (theta + 45) / 180 * np.pi
    else:
        alpha_2 = (theta * -1 + 45) / 180 * np.pi

    theta_aff = np.zeros((2, 3), dtype=np.float32)
    theta_aff[0, 2] = 0
    theta_aff[1, 2] = 0
    theta_aff[0, 0] = np.cos(alpha)
    theta_aff[0, 1] = (-np.sin(alpha))
    theta_aff[1, 0] = np.sin(alpha)
    theta_aff[1, 1] = np.cos(alpha)

    x = 0.5 * size * (np.sqrt(2) * np.sin(alpha_2) - 1) / np.sin(alpha_2)
    y = size * (np.sin(alpha_2 - np.pi / 4) * np.cos(alpha_2 - np.pi / 4)) / np.sin(alpha_2)
    t = 0.5 * size / np.sin(alpha_2)

    scale = (x + t) / (y + t)

    theta_aff = torch.from_numpy(theta_aff * scale).unsqueeze(0)

    return theta_aff, theta, np.cos(alpha_2)
