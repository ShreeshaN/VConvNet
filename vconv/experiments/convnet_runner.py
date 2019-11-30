# -*- coding: utf-8 -*-
"""
@created on: 11/24/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

from vconv.networks.conv_network import SmallConvNet, VariableConvNet
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import os
import cv2
import numpy as np
from torch import tensor
import time
import json
from vconv.utils import file_utils


class ConvNetRunner:
    def __init__(self, args):
        self.run_name = args.run_name + '_' + str(time.time()).split('.')[0]
        self.current_run_basepath = args.network_metrics_basepath + '/' + self.run_name + '/'
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.test_net = args.test_net
        self.train_net = args.train_net
        self.batch_size = args.batch_size
        classes = list(args.class_mappings)
        self.class_mappings = {class_: i for i, class_ in enumerate(classes)}
        self.num_classes = len(classes)
        self.variable_net = args.variable_net
        self.images_basepath = args.images_basepath
        self.train_data_file = args.train_data_file
        self.test_data_file = args.test_data_file
        self.is_cuda_available = torch.cuda.is_available()
        self.display_interval = args.display_interval

        self.network_metrics_basepath = args.network_metrics_basepath
        self.tensorboard_summary_path = self.current_run_basepath + args.tensorboard_summary_path
        self.network_save_path = self.current_run_basepath + args.network_save_path

        self.network_restore_path = args.network_restore_path

        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        self.network_save_interval = args.network_save_interval
        self.normalise = args.normalise
        self.dropout = args.dropout

        paths = [self.network_save_path, self.tensorboard_summary_path]
        file_utils.create_dirs(paths)

        self.network = None
        if self.variable_net:
            self.network = VariableConvNet(self).to(self.device)
        else:
            self.network = SmallConvNet(self).to(self.device)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimiser = optim.Adam(self.network.parameters(), lr=args.learning_rate)

        if self.train_net:
            self.network.train()
            self.log_file = open(self.network_save_path + '/' + self.run_name + '.log', 'w')
            self.log_file.write(json.dumps(args))
        if self.test_net:
            print('Loading Network')
            self.network.load_state_dict(torch.load(self.network_restore_path, map_location=self.device))
            self.network.eval()
            self.log_file = open(self.network_restore_path + '/' + self.run_name + '.log', 'a')
            print('\n\n\n********************************************************', file=self.log_file)
            print('Testing Model - ', self.network_restore_path)
            print('Testing Model - ', self.network_restore_path, file=self.log_file)
            print('********************************************************', file=self.log_file)

        self.writer = SummaryWriter(self.tensorboard_summary_path)

        self.batch_loss, self.batch_accuracy = [], []

    def read_images(self, files):
        return [cv2.imread(self.images_basepath + file, 0) for file in files]
        # return tensor([cv2.imread(self.images_basepath + file, 0) for file in files]).to(self.device).float().unsqueeze(
        #         1)

    def log_summary(self, global_step, tr_accuracy, tr_loss, te_accuracy, te_loss):
        self.writer.add_scalar('Train/Epoch Accuracy', tr_accuracy, global_step)
        self.writer.add_scalar('Train/Epoch Loss', tr_loss, global_step)
        self.writer.add_scalar('Test/Accuracy', te_accuracy, global_step)
        self.writer.add_scalar('Test/Loss', te_loss, global_step)
        self.writer.flush()

    def data_reader(self, data_file, should_batch=True, shuffle=True, normalise=False):
        data = pd.read_csv(data_file)
        data = data.apply(lambda x: x.str.strip(), axis=1)  # Removing spaces from data
        if shuffle:
            data = data.sample(frac=1)
        input_data, labels = self.read_images(data['Image_name'].values), self.get_class_mappings(data['Label'].values)
        if normalise:
            input_data = self.normalise_data(input_data)
        if should_batch:
            batched_input = [input_data[pos:pos + self.batch_size] for pos in
                             range(0, len(data), self.batch_size)]
            batched_labels = [labels[pos:pos + self.batch_size] for pos in range(0, len(data), self.batch_size)]
            return batched_input, batched_labels
        else:
            return input_data, labels

    def get_onehot(self, class_):
        label = np.zeros(self.num_classes)
        label[self.class_mappings[class_]] = 1
        return label

    def get_class_mappings(self, labels):
        return tensor([self.class_mappings[x] for x in labels]).to(self.device)

    def normalise_data(self, data):
        data = np.asarray(data)
        temp = [np.max(data), np.min(data)]
        diff = temp[0] - temp[1]
        return (data - temp[1]) / diff

        # temp = [torch.max(data), torch.min(data)]
        # diff = temp[0] - temp[1]
        # return (data - temp[1]) / diff

    # def get_multiclass_accuracy(self, predictions, label):
    #     total = len(predictions)
    #     return len([0 for pred, l in zip(predictions, label) if np.argmax(softmax(pred)) == np.argmax(l)]) / total * 100

    def train(self):
        train_data, train_labels = self.data_reader(self.train_data_file, normalise=self.normalise)
        test_data, test_labels = self.data_reader(self.test_data_file, should_batch=False, shuffle=False,
                                                  normalise=self.normalise)
        total_step = len(train_data)

        for epoch in range(1, self.epochs):
            self.batch_loss, self.batch_accuracy = [], []
            for i, (image_data, label) in enumerate(zip(train_data, train_labels)):
                predictions = self.network(image_data)
                loss = self.loss_function(predictions, label)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                predictions = nn.Softmax()(predictions)
                _, predictions = torch.max(predictions.data, 1)
                correct = (predictions == label).sum().item()
                accuracy = correct / label.size(0)
                self.batch_loss.append(loss.detach().numpy())
                self.batch_accuracy.append(accuracy)
                if i % self.display_interval == 0:
                    print(
                            f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | Loss: {loss} | Accuracy: {accuracy}")
                    print(
                            f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | Loss: {loss} | Accuracy: {accuracy}",
                            file=self.log_file)

            # Test data
            with torch.no_grad():
                test_predictions = self.network(test_data)
                test_loss = self.loss_function(test_predictions, test_labels)
                test_predictions = nn.Softmax()(test_predictions)
                _, test_predictions = torch.max(test_predictions.data, 1)
                correct = (test_predictions == test_labels).sum().item()
                print('***** Test Metrics ***** ')
                print('***** Test Metrics ***** ', file=self.log_file)
            test_accuracy = correct / test_labels.size(0)
            print(f"Loss: {test_loss} | Accuracy: {test_accuracy}")
            print(f"Loss: {test_loss} | Accuracy: {test_accuracy}", file=self.log_file)

            self.log_summary(epoch, np.mean(self.batch_accuracy), np.mean(self.batch_loss), test_accuracy,
                             test_loss.numpy())

            if epoch % self.network_save_interval == 0:
                save_path = self.network_save_path + '/' + self.run_name + '_' + str(epoch) + '.pt'
                torch.save(self.network.state_dict(), save_path)
                print('Network successfully saved: ' + save_path)

    def test(self):
        test_data, test_labels = self.data_reader(self.train_data_file, should_batch=False, shuffle=False,
                                                  normalise=self.normalise())
        test_predictions = self.network(test_data).detach()
        test_predictions = nn.Softmax()(test_predictions)
        _, test_predictions = torch.max(test_predictions.data, 1)
        correct = (test_predictions == test_labels).sum().item()
        test_accuracy = correct / test_labels.size(0)
        print(correct, test_labels.size(0))
        print(f"Accuracy: {test_accuracy}")
        print(f"Accuracy: {test_accuracy}", file=self.log_file)
