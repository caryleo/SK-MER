'''
Description: 
Autor: Gary Liu
Date: 2021-07-02 11:55:01
LastEditors: Gary Liu
LastEditTime: 2021-12-10 16:48:57
'''
import argparse
from parse_config import ConfigParser
import collections
import torch
import random
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
from utils.util import create_model, create_dataloader, create_trainer, create_trainer_cl

# fix random seeds for reproducibility
SEED = 125
# print("Seeting random seed as", SEED)
random.seed(SEED)
torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.enabled = True 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# torch.autograd.set_detect_anomaly(True)

torch.set_num_threads(1)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = create_dataloader(config)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = create_model(config)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    print("!!! Caution! Training Mode:", config['mode'])
    if config['mode'] == 'cl':
        criterion_cl = getattr(module_loss, config['loss_cl'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    if config['mode'] == 'cl':
        trainer = create_trainer_cl(model, criterion, metrics, logger,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      criterion_cl= criterion_cl
                      )
    else:
        trainer = create_trainer(model, criterion, metrics, logger,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader
                      )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Emotion Reasoning in Daily Life')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    main(config)
