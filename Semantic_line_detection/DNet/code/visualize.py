import cv2

import matplotlib.pyplot as plt

from libs.utils import *
from libs.modules import *

class Visualize_plt(object):

    def __init__(self, cfg=None):

        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width
        self.size = to_tensor(np.float32(cfg.size))

        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255

        self.param = {'linewidth': [0, 1, 2, 3, 4], 'color': ['yellow', 'red', 'lime']}

        self.show = {}


    def load_image(self, dir_name, file_name, name):
        # org image
        img = cv2.imread(dir_name + file_name)
        img = cv2.resize(img, (self.height, self.width))
        self.show[name] = img
        self.img = img

    def show_image(self):
        plt.figure()
        plt.imshow(self.img[:, :, [2, 1, 0]])

    def save_fig(self, path, name):
        mkdir(path)
        plt.axis('off')
        plt.savefig(path + name, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_img(self, dir_name, file_name, list):
        disp = self.line
        for i in range(len(list)):
            disp = np.concatenate((disp, self.show[list[i]], self.line), axis=1)

        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name, disp)


    def draw_lines_plt(self, pts, idx1, idx2, linestyle='-', zorder=1):
        endpts = find_endpoints(pts, [self.width-1, self.height-1])

        pt_1 = (endpts[0], endpts[1])
        pt_2 = (endpts[2], endpts[3])
        plt.plot([pt_1[0], pt_2[0]], [pt_1[1], pt_2[1]],
                 linestyle=linestyle,
                 linewidth=self.param['linewidth'][idx1],
                 color=self.param['color'][idx2],
                 zorder=zorder)

    def draw_lines_cv(self, data, name, ref_name='img', color=(255, 0, 0), s=1):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        for i in range(data.shape[0]):
            pts = data[i]
            pt_1 = (pts[0], pts[1])
            pt_2 = (pts[2], pts[3])
            img = cv2.line(img, pt_1, pt_2, color, s)

        self.show[name] = img

    def display_for_train_detector(self, batch, out, idx):
        img_name = batch['img_name'][0]
        candidates = batch['train_data'][0].cuda()
        self.show['img_name'] = img_name

        img = to_np(batch['img_rgb'][0].permute(1, 2, 0))
        img = np.uint8(img[:, :, [2, 1, 0]] * 255)
        self.show['img'] = img

        pos_line = candidates[candidates[:, 5] == 1][:, :4]
        neg_line = candidates[candidates[:, 5] == 0][:, :4]

        cls_out = torch.argmax(out['cls'], dim=1)
        out_line = candidates[cls_out == 1][:, :4] + \
                   out['reg'][cls_out == 1] * self.size

        pos_line = to_np(pos_line)
        neg_line = to_np(neg_line)
        out_line = to_np2(out_line)

        self.draw_lines_cv(data=pos_line, name='pos')
        self.draw_lines_cv(data=neg_line, name='neg')
        self.draw_lines_cv(data=out_line, name='out')

        self.save_img(dir_name=self.cfg.output_dir + 'train/display/',
                      file_name=str(idx)+'.jpg',
                      list=['pos', 'neg', 'out'])

    def display_for_test(self, batch, out):
        img_name = batch['img_name'][0]
        self.load_image(dir_name=self.cfg.img_dir,
                        file_name=img_name,
                        name='img')
        self.show_image()

        if 'pri' in out.keys():
            pts_pri = to_np(out['pri'])
            self.draw_lines_plt(pts_pri[0], 4, 1, '--', zorder=2)
        if 'mul' in out.keys():
            pts_mul = to_np(out['mul'])
            for i in range(pts_mul.shape[0]):
                self.draw_lines_plt(pts_mul[i], 4, 0, '-', zorder=1)

        self.save_fig(path=self.cfg.output_dir + 'test/out_single/',
                      name=img_name[:-4] + '.png')

