
import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from create_dataset import *
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data.sampler as sampler


class VGG16(nn.Module):
    def __init__(self, device):
        """
        VGG16 net implementation.
        """
        super(VGG16, self).__init__()

        # use GPU, if available
        self.device = device
        self.to(self.device)

        # rebuilding VGGnet
        filter = [64, 128, 256, 512, 512]
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        # TODO Softmax?
        self.classifier = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], 20),  # TODO 20 has to be changed depending on number of output classes? -> make hyperparameter?
        )

        # weight and bias initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)



    def conv_layer(self, in_channel, out_channel, index):
        """
        Defining one convolution block used in VGG net.

        :param in_channel: Number of input channels
        :param out_channel: Number of output channels
        :param index: TODO
        :return:
        """
        if index < 3:
            # (Conv2D, BatchNorm, ReLu)x2, MaxPool
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            # (Conv2D, BatchNorm, ReLu)x3, MaxPool
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        return conv_block


    def forward(self, x):
        """
        Forward pass function (compare Pytorch API).

        :param x: image (3 dimensions?)
        :return:
        """
        g_block1 = self.block1(x)
        g_block2 = self.block2(g_block1)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        g_block5 = self.block5(g_block4)

        t1_pred = self.classifier(g_block5.view(g_block5.size(0), -1))
        t1_pred = F.softmax(t1_pred, dim=1)

        return t1_pred


    def compute_focal_loss(self, pred, label):
        """
        Compute focal loss.
        :param pred: one-hot prediction vector
        :param label: class index label vector (NOT one hot!)
        :return: scalar (as Tensor) of averaged loss over minibatch
        """
        print("in focal loss: ", label)
        # make label onehot TODO BAD STYLE -> is not actually part of the loss computation, but left for now, because auxiliary complicated
        label_onehot = torch.zeros(pred.size()).to(self.device)
        label_onehot.scatter_(1, label.unsqueeze(1), 1).to(self.device)

        # normal cross entropy
        # loss = x_output_onehot * torch.log(x_pred + 1e-20)
        # focal loss
        loss = - (label_onehot * ((1 - pred) ** 2) * torch.log(pred + 1e-20)).to(self.device)  # size: (config['batch_size'], principal_classes), e.g. (100, 20)
        loss = torch.sum(loss, dim=1)  # sum over the classes dimension -> per instance loss
        loss = torch.mean(loss)  # average loss over the batch

        return loss
