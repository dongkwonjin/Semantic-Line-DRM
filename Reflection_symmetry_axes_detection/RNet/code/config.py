import os

class Config(object):
    def __init__(self):

        # proj & output dir
        self.proj_dir = os.path.dirname(os.getcwd()) + '/'
        self.output_dir = self.proj_dir + 'output_v2/'

        # dataset dir

        self.train_dataset_dir = '/media/dkjin/3535ee90-f88e-4f09-b6aa-cfbf15169dde' \
                                 '/Github/Semantic-Line-DRM/Dataset/ICCV_2017/'
        self.train_img_dir = self.train_dataset_dir + 'train_images/'

        self.test_dataset = 'ICCV_2017'  # ['ICCV_2017', 'NYU', 'SYM_Hard]
        if self.test_dataset == 'ICCV_2017':
            self.test_dataset_dir = '/media/dkjin/3535ee90-f88e-4f09-b6aa-cfbf15169dde' \
                                    '/Github/Semantic-Line-DRM/Dataset/ICCV_2017/'

            self.test_img_dir = self.test_dataset_dir + 'test_images/'
        if self.test_dataset == 'NYU':
            self.test_dataset_dir = '/media/dkjin/3535ee90-f88e-4f09-b6aa-cfbf15169dde' \
                                    '/Github/Semantic-Line-DRM/Dataset/NYU/'
            self.test_img_dir = self.test_dataset_dir
        if self.test_dataset == 'SYM_Hard':
            self.test_dataset_dir = '/media/dkjin/3535ee90-f88e-4f09-b6aa-cfbf15169dde' \
                                    '/Github/Semantic-Line-DRM/Dataset/SYM_Hard/'
            self.test_img_dir = self.test_dataset_dir + 'images/'

        # other dir
        self.weight_dir = self.output_dir + '/train/weight/'
        self.paper_weight_dir = "/".join(self.proj_dir.split("/")[:-2]) + '/paper_weight/'
        self.pretrained_DNet_dir = "/".join(self.proj_dir.split("/")[:-2]) + '/DNet/output/train/weight/'

        # setting for train & test
        self.run_mode = 'test_paper'  # ['train', 'test', 'test_paper']
        self.resume = True

        self.gpu_id = "0"
        self.seed = 123
        self.num_workers = 4
        self.epochs = 500
        self.ratio_pos = 0.4
        self.batch_size = {'img': 1,
                           'train_line': 100,
                           'train_pair': 30,
                           'test_line': 100}

        # optimizer
        self.lr = 1e-4
        self.milestones = [50, 100, 150, 200]
        self.weight_decay = 5e-4
        self.gamma = 0.5

        # other setting
        self.height = 400
        self.width = 400
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
