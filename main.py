import os
import yaml
from lib.function import Trainer, Tester


if __name__ == '__main__':
    cfg_file = open('configs/config.yaml', 'r')
    cfg = yaml.load(cfg_file)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['CUDA_VISIBLE_DEVICES']

    if cfg['MODE'] == 'train':
        trainer = Trainer(cfg)
        trainer.train()
    elif cfg['MODE'] == 'test':
        tester = Tester(cfg)
        tester.test()


