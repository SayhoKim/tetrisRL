import argparse
import os
import yaml
from lib.function import Trainer, Tester


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/config.yaml', help='Configuration File Path')
    parser.add_argument('--train', type=bool, default=True, help='Training Mode')
    opt = parser.parse_args()

    cfg_file = open(opt.cfg, 'r')
    cfg = yaml.load(cfg_file)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['CUDA_VISIBLE_DEVICES']

    if opt.train:
        trainer = Trainer(cfg)
        trainer.train()
    else:
        tester = Tester(cfg)
        tester.test()
