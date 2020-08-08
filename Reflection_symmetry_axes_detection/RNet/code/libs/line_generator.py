import numpy as np

from libs.modules import *

class Training_Line_Generator(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.thresd_overlap = 0.4

        self.height = cfg.height
        self.width = cfg.width


        # outlier threshold
        self.thresd_l = 150  # threshold of length
        self.thresd_b = 30  # threshold of distance from image boundary
        self.interval = 15  # interval pixel of sampling
        self.pos_interval = 10

        # image boundary coordinates
        grid_X, grid_Y = np.linspace(0, self.width - 1, self.width), \
                         np.linspace(0, self.height - 1, self.height)

        grid_X = np.expand_dims(grid_X, axis=1)  # W
        grid_Y = np.expand_dims(grid_Y, axis=1)  # H
        self.sample_num = grid_X.shape[0]

        temp_H1 = np.zeros((self.height, 1), dtype=np.float32)
        temp_H2 = np.zeros((self.height, 1), dtype=np.float32)
        temp_W1 = np.zeros((self.width, 1), dtype=np.float32)
        temp_W2 = np.zeros((self.width, 1), dtype=np.float32)
        temp_H2[:] = self.width - 1
        temp_W2[:] = self.height - 1

        self.line = {}
        self.line[0] = np.concatenate((grid_X, temp_W1), axis=1)
        self.line[1] = np.concatenate((temp_H1, grid_Y), axis=1)
        self.line[2] = np.concatenate((grid_X, temp_W2), axis=1)
        self.line[3] = np.concatenate((temp_H2, grid_Y), axis=1)

        self.dx = np.array([1, 0, -1, 0], dtype=np.float32)
        self.dy = np.array([0, -1, 0, 1], dtype=np.float32)

        self.div = 2

        self.candidate_line()

    def candidate_line(self):

        cand_line = []

        for i in range(4):
            for j in range(i+1, 4):

                # select two side in image boundary
                ref_1 = self.line[i]
                ref_2 = self.line[j]

                for p1 in range(1, self.height - 1, self.interval):
                    for p2 in range(1, self.width - 1, self.interval):

                        # select two endpoints in two sides respectively
                        pt_1 = ref_1[p1]
                        pt_2 = ref_2[p2]

                        endpts = np.concatenate((pt_1, pt_2), axis=0)
                        check = candidate_line_filtering(pts=endpts,
                                                         size=(self.height, self.width),
                                                         thresd_boundary=self.thresd_b,
                                                         thresd_length=self.thresd_l)

                        if check == 0:
                            cand_line.append(np.expand_dims(endpts, axis=0))

        cand_line = np.float32(np.concatenate(cand_line))
        self.candidates = cand_line

        return cand_line

    def negative_line(self):

        total = {'neg': []}
        step = 10
        cxy = np.zeros((step, 2), dtype=np.float32)

        cx = np.linspace(self.gtline[0, 0], self.gtline[0, 2], step)
        cy = np.linspace(self.gtline[0, 1], self.gtline[0, 3], step)
        cxy[:, 0] = cx
        cxy[:, 1] = cy

        a = (self.gtline[:, 1] - self.gtline[:, 3]) / (self.gtline[:, 0] - self.gtline[:, 2])
        angle = np.arctan(a) / np.pi * 180

        data = []
        for i in range(step):
            pt = np.array([cxy[i, 0], cxy[i, 1]])
            for theta in range(0, 180, 10):

                if np.abs(theta - angle) < 10:
                    continue
                theta_t = theta / 180 * np.pi

                dx = np.cos(theta_t)
                dy = np.sin(theta_t)
                mpt = pt + np.float32([dx, dy])

                endpts = np.concatenate((pt, mpt))
                endpts = find_endpoints(endpts, (399, 399, 3))

                if endpts.shape[0] != 4:
                    continue

                check = candidate_line_filtering(pts=endpts,
                                                 size=(self.height, self.width),
                                                 thresd_boundary=self.thresd_b,
                                                 thresd_length=self.thresd_l)
                if check == 0:
                    data.append(endpts[:4])

        p = (-1) * 1 / a

        for offset in range(-50, 50, 15):
            dx = np.cos(p) * offset
            dy = np.sin(p) * offset

            txy = cxy + np.array([dx, dy]).transpose(1, 0)

            for i in range(step):
                pt = np.array([txy[i, 0], txy[i, 1]])
                for theta in range(0, 180, 10):
                    if np.abs(theta - angle) < 10:
                        continue

                    theta_t = theta / 180 * np.pi

                    dx = np.cos(theta_t)
                    dy = np.sin(theta_t)
                    mpt = pt + np.float32([dx, dy])

                    endpts = np.concatenate((pt, mpt))
                    endpts = find_endpoints(endpts, (self.width - 1, self.height - 1))

                    if endpts.shape[0] != 4:
                        continue

                    check = candidate_line_filtering(pts=endpts,
                                                     size=(self.height, self.width),
                                                     thresd_boundary=self.thresd_b,
                                                     thresd_length=self.thresd_l)
                    if check == 0:
                        data.append(endpts[:4])

        total['neg'].append(np.array(data))

        return total

    def positive_line(self):

        num, _ = self.candidates.shape

        data = {'pos': []}

        for i in range(self.gtline.shape[0]):

            temp = {'endpts': []}

            for j in range(4):
                endpts = []

                # two endpoints
                if j < 2:
                    x = self.gtline[i, 0]
                    y = self.gtline[i, 1]
                    endpts.append(np.copy(self.gtline[i, :2]))
                else:
                    x = self.gtline[i, 2]
                    y = self.gtline[i, 3]
                    endpts.append(np.copy(self.gtline[i, 2:]))

                # set table
                if j % 2 == 0:
                    visit = np.zeros((self.height, self.width), dtype=np.int32)
                    visit[1:self.height - 1, 1:self.width - 1] = 1

                visit[int(y), int(x)] = 1

                for k in range(0, self.pos_interval):
                    for l in range(4):
                        nx = x + self.dx[l]
                        ny = y + self.dy[l]
                        if ny < 0 or nx< 0 or ny >= self.height or nx >= self.width:
                            continue
                        if visit[int(ny), int(nx)] == 0:
                            x = nx
                            y = ny
                            visit[int(y), int(x)] = 1
                            break
                    endpts.append(np.array([x, y]))

                temp['endpts'].append(endpts)

            endpts_1 = np.concatenate(temp['endpts'][:2], axis=0)
            endpts_2 = np.concatenate(temp['endpts'][2:], axis=0)

            pos_line = []
            for p1 in range(0, self.pos_interval * 2 + 1):
                for p2 in range(0, self.pos_interval * 2 + 1):
                    pos_line.append(np.concatenate((endpts_1[p1], endpts_2[p2]), axis=0))

            data['pos'].append(np.float32(pos_line))


        return data

    def update(self, img_path=None, gtline=None):

        if img_path is not None:
            self.img_path = img_path
        if gtline is not None:
            self.gtline = gtline


def candidate_line_filtering(pts, size, thresd_boundary, thresd_length):
    ## exclude outlier -> short length, boundary
    check = 0

    pt_1 = pts[:2]
    pt_2 = pts[2:]

    length = np.sqrt(np.sum(np.square(pt_1 - pt_2)))

    # short length
    if length < thresd_length:
        check += 1

    # boundary
    if (pt_1[0] < thresd_boundary and pt_2[0] < thresd_boundary):
        check += 1
    if (pt_1[1] < thresd_boundary and pt_2[1] < thresd_boundary):
        check += 1
    if (abs(pt_1[0] - size[0]) < thresd_boundary and abs(pt_2[0] - size[0]) < thresd_boundary):
        check += 1
    if (abs(pt_1[1] - size[1]) < thresd_boundary and abs(pt_2[1] - size[1]) < thresd_boundary):
        check += 1
    return check

