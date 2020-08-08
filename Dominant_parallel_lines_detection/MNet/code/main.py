from config import Config
from test import *
from train import *
from libs.prepare import *

def main_test(cfg, dict_DB):

    # test optoin
    if cfg.test_mode == 'vanishing_point':
        test_process = Test_Process_DRM_vp(cfg, dict_DB)
    elif cfg.test_mode == 'vanishing_line':
        test_process = Test_Process_DRM_line(cfg, dict_DB)

    test_process.run(dict_DB['DNet'], dict_DB['RNet'], dict_DB['MNet'])


def main_train(cfg, dict_DB):

    if cfg.test_mode == 'vanishing_point':
        dict_DB['test_process'] = Test_Process_DRM_vp(cfg, dict_DB)
    elif cfg.test_mode == 'vanishing_line':
        dict_DB['test_process'] = Test_Process_DRM_line(cfg, dict_DB)

    train_process = Train_Process_MNet(cfg, dict_DB)
    train_process.run()


def main():

    # Config
    cfg = Config()

    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    torch.backends.cudnn.deterministic = True

    # prepare
    dict_DB = dict()
    dict_DB = prepare_visualization(cfg, dict_DB)
    dict_DB = prepare_dataloader(cfg, dict_DB)
    dict_DB = prepare_model(cfg, dict_DB)
    dict_DB = prepare_postprocessing(cfg, dict_DB)
    dict_DB = prepare_training(cfg, dict_DB)

    if 'test' in cfg.run_mode:
        main_test(cfg, dict_DB)
    if 'train' in cfg.run_mode:
        main_train(cfg, dict_DB)

if __name__ == '__main__':
    main()