# -*- coding: utf-8 -*-
"""
@created on: 11/24/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

from vconv.networks.ConvNetwork import SmallConvNet, VariableConvNet
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import os
import cv2
import numpy as np
from torch import tensor
from scipy.special import softmax


class ConvNetRunner:
    def __init__(self, args):
        self.run_name = args.run_name
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.test_net = args.test_net
        self.train_net = args.train_net
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.variable_net = args.variable_net
        self.train_data_file = args.train_data_file
        self.test_data_file = args.test_data_file
        self.is_cuda_available = torch.cuda.is_available()
        self.tensorboard_summary_path = args.tensorboard_summary_path
        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        self.network_restore_path = args.network_restore_path
        self.network_save_path = args.network_save_path
        self.class_mappings = args.class_mappings
        self.network_save_interval = args.network_save_interval
        self.create_dirs()

        self.network = None
        if self.variable_net:
            self.network = VariableConvNet(self.num_classes).to(self.device)
        else:
            self.network = SmallConvNet(self.num_classes).to(self.device)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimiser = optim.Adam(self.network.parameters(), lr=args.learning_rate)

        if self.train_net:
            self.network.train()
            self.log_file = open(self.network_save_path + '/' + self.run_name + '.log', 'w')
        if self.test_net:
            print('Loading Network')
            self.network.load_state_dict(torch.load(self.network_restore_path, map_location=self.device))
            self.network.eval()
            self.log_file = open(self.network_save_path + '/' + self.run_name + '.log', 'a')
            print('\n\n\n********************************************************', file=self.log_file)
            print('Testing Model', file=self.log_file)
            print('********************************************************', file=self.log_file)

        self.writer = SummaryWriter(args.tensorboard_summary_path)

        self.batch_loss, self.batch_accuracy = [], []

    def create_dirs(self):
        paths = [self.network_save_path, self.tensorboard_summary_path]
        [os.makedirs(path) for path in paths if not os.path.exists(path)]

    def read_images(self, files):
        return tensor([cv2.imread(file, 0) for file in files]).to(self.device).float().unsqueeze(
                1)

    def log_summary(self, global_step, tr_accuracy, tr_loss, te_accuracy, te_loss):
        self.writer.add_scalar('Train/Epoch Accuracy', tr_accuracy, global_step)
        self.writer.add_scalar('Train/Epoch Loss', tr_loss, global_step)
        self.writer.add_scalar('Test/Accuracy', te_accuracy, global_step)
        self.writer.add_scalar('Test/Loss', te_loss, global_step)
        self.writer.flush()

    def data_reader(self, data_file, should_batch=True, shuffle=True):
        data = pd.read_csv(data_file)
        data = data.apply(lambda x: x.str.strip(), axis=1)  # Removing spaces from data
        if shuffle:
            data = data.sample(frac=1)
        input_data, labels = self.read_images(data['Images'].values), data['Label'].values
        if should_batch:
            batched_input = [input_data[pos:pos + self.batch_size] for pos in
                             range(0, len(data), self.batch_size)]
            batched_labels = [labels[pos:pos + self.batch_size] for pos in range(0, len(labels), self.batch_size)]
            return batched_input, batched_labels
        else:
            return input_data, labels

    def get_onehot(self, class_):
        label = np.zeros(self.num_classes)
        label[self.class_mappings[class_]] = 1
        return label

    # def get_multiclass_accuracy(self, predictions, label):
    #     total = len(predictions)
    #     return len([0 for pred, l in zip(predictions, label) if np.argmax(softmax(pred)) == np.argmax(l)]) / total * 100

    def train(self):
        train_data, train_labels = self.data_reader(self.train_data_file)
        test_data, test_labels = self.data_reader(self.test_data_file, should_batch=False, shuffle=False)
        total_step = len(train_data)

        for epoch in range(1, self.epochs):
            self.batch_loss, self.batch_accuracy = [], []
            for i, (image_data, label) in enumerate(zip(train_data, train_labels)):
                label = tensor([self.class_mappings[x] for x in label]).to(self.device)
                predictions = self.network(image_data)
                loss = self.loss_function(predictions, label)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                _, predictions = torch.max(predictions.data, 1)
                correct = (predictions == label).sum().item()
                accuracy = correct / label.size(0)
                self.batch_loss.append(loss.detach().numpy())
                self.batch_accuracy.append(accuracy)
                print(f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | Loss: {loss} | Accuracy: {accuracy}")
                print(f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | Loss: {loss} | Accuracy: {accuracy}",
                      file=self.log_file)

            # Test data
            with torch.no_grad():
                if epoch == 1:
                    test_labels = tensor([self.class_mappings[x] for x in test_labels]).to(self.device)
                test_predictions = self.network(test_data)
                test_loss = self.loss_function(test_predictions, test_labels)
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
        test_data, test_labels = self.data_reader(self.test_data_file, should_batch=False, shuffle=False)
        pass
