import yaml
from lib.trainer import Trainer


if __name__ == '__main__':
    cfg_file = open('configs/config.yaml', 'r')
    cfg = yaml.load(cfg_file)
    trainer = Trainer(cfg['CONFIG'])
    trainer.train()

