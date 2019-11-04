import argparse
import os
import yaml

from keras import backend as K
from lib.function import Trainer, Tester


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/config.yaml', help='Configuration File Path')
    parser.add_argument('--demo', action='store_true', default=False, help='Demo Mode')
    opt = parser.parse_args()

    cfg_file = open(opt.cfg, 'r')
    cfg = yaml.load(cfg_file)
    if len(K.tensorflow_backend._get_available_gpus()) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in cfg['NUM_GPUS']])

    if not opt.demo:
        trainer = Trainer(cfg)
        trainer.train()
    else:
        tester = Tester(cfg)
        tester.test()
