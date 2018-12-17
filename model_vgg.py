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

from config import config


class VGG16(nn.Module):
    def __init__(self):
        """
        VGG16 net implementation.
        """
        super(VGG16, self).__init__()

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


    # TODO
    def compute_loss(self, x_pred, x_output, num_output):
        """
        TODO

        :param x_pred:
        :param x_output:
        :param num_output:
        :return:
        """
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # loss = x_output_onehot * torch.log(x_pred + 1e-20) # normal cross entropy
        # focal loss
        loss = - (x_output_onehot * ((1 - x_pred)**2) * torch.log(x_pred + 1e-20))  # size: (config['batch_size'], principal_classes)
        # print("loss: ", loss, loss.size())
        return torch.sum(loss, dim=1)  # why summing here?


if __name__ == '__main__':

    # instantiating some variables
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']

    # load CIFAR100 dataset  # TODO
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # ?
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

    ])
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

    ])

    data_path = '/Users/FabianFalck/Documents/[03]SelfAuxLearning/Data'

    cifar100_train_set = CIFAR100(data_path=data_path, train=True, transform=trans_train)
    cifar100_test_set = CIFAR100(data_path=data_path, train=False, transform=trans_test)

    class ConcatDataset(torch.utils.data.Dataset):
        def __init__(self, *datasets):
            self.datasets = datasets

        def __getitem__(self, i):
            return tuple(d[i] for d in self.datasets)

        def __len__(self):
            return min(len(d) for d in self.datasets)


    kwargs = {'num_workers': 1, 'pin_memory': True}
    cifar100_train_loader = torch.utils.data.DataLoader(
        dataset=cifar100_train_set,
        batch_size=batch_size,
        sampler=sampler.RandomSampler(np.arange(0, 50000)))

    cifar100_test_loader = torch.utils.data.DataLoader(
        dataset=cifar100_test_set,
        batch_size=batch_size,
        shuffle=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use GPU or CPU, whatever is available
    VGG16 = VGG16().to(device)
    optimizer = optim.SGD(VGG16.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # learning rate decay by factor 0.5 every 50 training epochs (not (!) batches)

    # define parameters
    n_train_batches = len(cifar100_train_loader)  # epoch equals total number of batches
    n_test_batches = len(cifar100_test_loader)  # "test" epoch equals total number of batches
    k = 0
    avg_cost = np.zeros([n_epochs, 8], dtype=np.float32)  # TODO why 8?

    print("training started...")
    for epoch in range(n_epochs):
        cost = np.zeros(4, dtype=np.float32)  # TODO why 4?
        scheduler.step()  # update learning rate

        # new iteration over all training batches
        cifar100_train_dataset = iter(cifar100_train_loader)  # TODO ISSUE: sequence of traversing through training data will be the same in every epoch
        for i in range(n_train_batches):
            # if i % 10 == 0:
            print("batch %d"%i)

            # evaluating training datata
            train_data, train_label, _ = cifar100_train_dataset.next()  # get next training batch
            train_label = train_label.type(torch.LongTensor)  # TODO why not done in loader itself?
            train_data, train_label = train_data.to(device), train_label.to(device)  # TODO why not done in loader itself?
            train_pred1 = VGG16(train_data)  # TODO WEIRD?
            optimizer.zero_grad()
            train_loss1 = VGG16.compute_loss(train_pred1, train_label[:, 2], 20)
            train_loss = torch.mean(train_loss1)

            train_loss.backward()  # backpropagate error
            optimizer.step()  # gradient step

            train_predict_label1 = train_pred1.data.max(1)[1]

            train_acc1 = train_predict_label1.eq(train_label[:, 2]).sum().item() / batch_size

            cost[0] = torch.mean(train_loss1).item()
            cost[1] = train_acc1
            k = k + 1
            avg_cost[epoch][0:4] += cost / n_train_batches

        # evaluating test data
        with torch.no_grad():
            cifar100_test_dataset = iter(cifar100_test_loader)
            for i in range(n_test_batches):
                test_data, test_label, _ = cifar100_test_dataset.next()
                test_label = test_label.type(torch.LongTensor)
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_pred1 = VGG16(test_data)
                test_loss1 = VGG16.compute_loss(test_pred1, test_label[:, 2], 20)

                test_predict_label1 = test_pred1.data.max(1)[1]

                test_acc1 = test_predict_label1.eq(test_label[:, 2]).sum().item() / batch_size
                cost[0] = torch.mean(test_loss1).item()
                cost[1] = test_acc1

                avg_cost[epoch][4:] += cost / n_test_batches

        print('Epoch: {:04d} Iteration: {:04d} | CIFAR100-TRAIN: {:.4f} {:.4f}  || '
              'CIFAR100-TEST: {:.4f} {:.4f} '
              .format(epoch, k, avg_cost[epoch][0], avg_cost[epoch][1],
                      avg_cost[epoch][4], avg_cost[epoch][5]))

    #torch.save(VGG16.state_dict(), 'model_weights/cifar10_human_old')
    #np.save('loss/cifar10_human_old.npy', avg_cost)
