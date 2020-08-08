import numpy as np

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

        # not include in positive line

        num, _ = self.candidates.shape

        neg_check = np.ones((num), dtype=np.int32)

        data = {}
        data['neg'] = {}
        data['neg']['endpts'] = []
        data['neg']['cls'] = []
        data['neg']['offset'] = []

        for i in range(num):

            pts = self.candidates[i]

            # exclude positive line
            for k in range(self.gtline.shape[0]):

                d1 = d2 = 99999
                d1 = np.minimum(d1, np.sum(np.abs(pts[:2] - self.gtline[k, :2])))
                d2 = np.minimum(d2, np.sum(np.abs(pts[2:4] - self.gtline[k, 2:4])))
                d1 = np.minimum(d1, np.sum(np.abs(pts[2:4] - self.gtline[k, :2])))
                d2 = np.minimum(d2, np.sum(np.abs(pts[:2] - self.gtline[k, 2:4])))

                if d1 <= self.pos_interval and d2 <= self.pos_interval:  # positive line
                    neg_check[i] = 0
                    break

        neg_line = self.candidates[neg_check == 1]

        neg_num = neg_line[:, :4].shape[0]

        data['neg']['endpts'] = np.float32(neg_line[:, :4])
        data['neg']['offset'] = np.repeat(np.float32(np.array([[0, 0, 0, 0]])), neg_num, axis=0)
        data['neg']['cls'] = np.repeat(np.float32(np.array([[1, 0]])), neg_num, axis=0)

        return data

    def positive_line(self):

        num, _ = self.candidates.shape

        data = {}
        data['pos'] = {}
        data['pos']['endpts'] = []
        data['pos']['cls'] = []
        data['pos']['offset'] = []

        for i in range(self.gtline.shape[0]):

            temp = {'endpts': [],
                    'offset': []}
            for j in range(4):
                endpts = []
                offset = []

                # two endpoints
                if j < 2:
                    x = self.gtline[i, 0]
                    y = self.gtline[i, 1]
                    endpts.append(np.copy(self.gtline[i, :2]))
                    offset.append(np.array([0, 0]))
                else:
                    x = self.gtline[i, 2]
                    y = self.gtline[i, 3]
                    endpts.append(np.copy(self.gtline[i, 2:]))
                    offset.append(np.array([0, 0]))

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

                    if j < 2:
                        offset.append(np.array([self.gtline[i, 0] - x, self.gtline[i, 1] - y]))
                    else:
                        offset.append(np.array([self.gtline[i, 2] - x, self.gtline[i, 3] - y]))

                temp['endpts'].append(endpts)
                temp['offset'].append(offset)

            endpts_1 = np.concatenate(temp['endpts'][:2], axis=0)
            endpts_2 = np.concatenate(temp['endpts'][2:], axis=0)
            offset_1 = np.concatenate(temp['offset'][:2], axis=0)
            offset_2 = np.concatenate(temp['offset'][2:], axis=0)

            pos_line = []
            offset = []
            for p1 in range(0, self.pos_interval * 2 + 1):
                for p2 in range(0, self.pos_interval * 2 + 1):
                    pos_line.append(np.concatenate((endpts_1[p1], endpts_2[p2]), axis=0))
                    offset.append(np.concatenate((offset_1[p1], offset_2[p2]), axis=0))

            pos_num = len(pos_line)

            cls = np.repeat(np.float32(np.array([[0, 1]])), pos_num, axis=0)

            pos_line = np.float32(pos_line)
            offset = np.float32(offset)
            cls = np.float32(cls)

            data['pos']['endpts'].append(pos_line)
            data['pos']['offset'].append(offset)
            data['pos']['cls'].append(cls)


        # dict to array
        if data['pos']['endpts'] != []:
            data['pos']['endpts'] = np.concatenate(data['pos']['endpts'])
            data['pos']['offset'] = np.concatenate(data['pos']['offset'])
            data['pos']['cls'] = np.concatenate(data['pos']['cls'])

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

