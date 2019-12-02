# -*- coding: utf-8 -*-
"""
@created on: 11/27/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import torch
from vconv.networks.conv_network import SmallConvNet, VariableConvNet
from vconv.utils.class_utils import AttributeDict
from vconv.utils.file_utils import create_dirs, cp_file
import json
import argparse
from torch import tensor
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import torch.nn as nn


class ConvVisualiser:
    def __init__(self, args):
        self.is_cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        self.variable_net = args.variable_net
        self.network = None
        classes = list(args.class_mappings)
        self.class_mappings = {class_: i for i, class_ in enumerate(classes)}
        self.reverse_class_mappings = {i: class_ for i, class_ in enumerate(classes)}
        self.num_classes = len(classes)
        self.network_restore_path = args.network_restore_path
        self.dropout = 0
        if self.variable_net:
            self.network = VariableConvNet(self).to(self.device)
        else:
            self.network = SmallConvNet(self).to(self.device)

        self.network_restore()

        self.visualiser_path = '/'.join(self.network_restore_path.split('/')[:-2]) + "/" + args.visualiser_folder_name
        create_dirs([self.visualiser_path])

        self.images_to_visualise = args.images_to_visualise

    def network_restore(self):
        print('***************************************************')
        print('Restoring Network - ', self.network_restore_path)
        print('***************************************************')
        self.network.load_state_dict(torch.load(self.network_restore_path, map_location=self.device))
        self.network.eval()

    def plot_activations(self, kernal, kernal_name):
        kernal = kernal.squeeze(0).detach().numpy()
        total_cols = 4
        total_rows = kernal.shape[0] // total_cols
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i, activation_map in enumerate(kernal):
            ax = fig.add_subplot(total_rows, total_cols, i + 1)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.imshow(activation_map, interpolation="nearest", cmap="hot")
        print('Saving kernal plots ', kernal_name)
        fig.savefig(self.image_folder + '/' + str(kernal_name) + '.jpg')

    def plot_kernals(self):
        convs = ['conv1.weight', 'conv2.weight', 'conv3.weight']
        for conv in convs:
            print('Plotting ', conv)
            kernal = self.network.state_dict()[conv]
            if conv != 'conv1.weight':
                print(kernal.numpy().shape)
                kernal = np.take(kernal.numpy(), 1, axis=0)
                print(kernal.shape)
            else:
                kernal = kernal.squeeze(1).detach().numpy()
            total_cols = 4
            total_rows = kernal.shape[0] // total_cols
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            for i, activation_map in enumerate(kernal):
                ax = fig.add_subplot(total_rows, total_cols, i + 1)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.imshow(activation_map, interpolation="nearest")
            print('Saving kernal plots ', conv)
            fig.savefig(self.visualiser_path + '/' + str(conv) + '.jpg')

    def run(self):
        with torch.no_grad():
            print(self.images_to_visualise + "/*")
            print(glob.glob(self.images_to_visualise + "/*"))
            for file in glob.glob(self.images_to_visualise + "/*"):
                print('Reading image ', file)
                image = cv2.imread(file, 0)
                image_name = file.split("/")[-1]
                self.image_folder = self.visualiser_path + "/" + image_name.split('.')[0]
                create_dirs([self.image_folder])
                cp_file(file, self.image_folder + '/' + image_name)
                kernals = self.network.forward_pass_kernels(image)
                for i, kernal in enumerate(kernals):
                    self.plot_activations(kernal, i)
                if self.variable_net:
                    prediction = torch.argmax(
                            nn.Softmax()(self.network(tensor(image).unsqueeze(0)))).item()
                else:
                    prediction = torch.argmax(
                            nn.Softmax()(self.network(tensor(image).unsqueeze(0).float()))).item()

                print('Prediction - ', self.reverse_class_mappings[prediction])
                print("**" * 50)


if __name__ == '__main__':
    def parse():
        parser = argparse.ArgumentParser(description="visualiser_configs")
        parser.add_argument('--configs_file', type=str)
        args = parser.parse_args()
        return args


    args = parse().__dict__
    configs = json.load(open(args['configs_file']))
    configs = {**configs, **args}
    configs = AttributeDict(configs)
    visualiser = ConvVisualiser(configs)
    visualiser.run()
    # visualiser.plot_kernals()
