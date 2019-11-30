# -*- coding: utf-8 -*-
"""
@created on: 11/24/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import torch
from vconv.experiments.convnet_runner import ConvNetRunner
from vconv.utils.class_utils import AttributeDict
import json
import argparse



def parse():
    parser = argparse.ArgumentParser(description="cvonv_configs")
    parser.add_argument('--train_net', type=bool)
    parser.add_argument('--test_net', type=bool)
    parser.add_argument('--configs_file', type=str)
    args = parser.parse_args()
    return args


def run(args):
    network = ConvNetRunner(args=args)
    if args.train_net:
        network.train()

    if args.test_net:
        network.test()


if __name__ == '__main__':
    args = parse().__dict__
    configs = json.load(open(args['configs_file']))
    configs = {**configs, **args}
    configs = AttributeDict(configs)
    run(configs)
