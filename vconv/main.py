# -*- coding: utf-8 -*-
"""
@created on: 11/24/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from vconv.experiments.ConvNetRunner import ConvNetRunner

import argparse

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse():
    parser = argparse.ArgumentParser(description="runner")
    parser.add_argument('--run_name', default='convnet', type=str)
    parser.add_argument('--epochs', default=5000000, type=int)
    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--variable_net', default=False, type=bool)
    parser.add_argument('--train_data_file',
                        default='/Users/badgod/badgod_documents/ImageDataset/same_size_input/csv/train.csv', type=str)
    parser.add_argument('--test_data_file',
                        default='/Users/badgod/badgod_documents/ImageDataset/same_size_input/csv/test.csv', type=str)
    parser.add_argument('--network_save_path', type=str, default='trained_models/', help='')
    parser.add_argument('--tensorboard_summary_path', type=str, default='tensorboard_summary', help='')
    parser.add_argument('--network_restore_path', type=str, default='trained_models/', help='')
    parser.add_argument('--train_net', type=bool)
    parser.add_argument('--test_net', type=bool)
    parser.add_argument('--class_mappings', default={'rose': 0, 'tulip': 1, 'daisy': 2, 'dandelion': 3, 'sunflower': 4})
    parser.add_argument('--network_save_interval', default=20, type=int)
    args = parser.parse_args()
    return args


def run(args):
    network = ConvNetRunner(args=args)
    if args.train_net:
        network.train()

    if args.test_net:
        network.test()


if __name__ == '__main__':
    args = parse()
    run(args)
