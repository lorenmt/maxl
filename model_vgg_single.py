from create_dataset import *

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        filter = [64, 128, 256, 512, 512]

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        # define fc-layers in VGG-16
        self.classifier = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], 20),
        )

        # apply weight initialisation
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
        if index < 3:
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
        g_block1 = self.block1(x)
        g_block2 = self.block2(g_block1)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        g_block5 = self.block5(g_block4)

        t1_pred = self.classifier(g_block5.view(g_block5.size(0), -1))
        t1_pred = F.softmax(t1_pred, dim=1)

        return t1_pred

    def model_fit(self, x_pred, x_output, num_output):
        # convert a single label into a one-hot vector
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # apply focal loss
        loss = x_output_onehot * (1 - x_pred) ** 2 * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)


# define image transformation
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])
trans_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])

# load CIFAR-100 dataset with batch-size 100
# set keyword download=True at the first time to download the dataset
cifar100_train_set = CIFAR100(root='dataset', train=True, transform=trans_train, download=False)
cifar100_test_set = CIFAR100(root='dataset', train=False, transform=trans_test, download=False)

batch_size = 100
kwargs = {'num_workers': 1, 'pin_memory': True}
cifar100_train_loader = torch.utils.data.DataLoader(
    dataset=cifar100_train_set,
    batch_size=batch_size,
    shuffle=True)

cifar100_test_loader = torch.utils.data.DataLoader(
    dataset=cifar100_test_set,
    batch_size=batch_size,
    shuffle=True)


# define VGG-16 model, and optimiser with learning rate 0.01, drop half for every 50 epochs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VGG16 = VGG16().to(device)
optimizer = optim.SGD(VGG16.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# define parameters and running for 200 epochs
total_epoch = 200
train_batch = len(cifar100_train_loader)
test_batch = len(cifar100_test_loader)
k = 0
avg_cost = np.zeros([total_epoch, 4], dtype=np.float32)
for index in range(total_epoch):
    cost = np.zeros(2, dtype=np.float32)

    # evaluate training data
    VGG16.train()
    cifar100_train_dataset = iter(cifar100_train_loader)
    for i in range(train_batch):
        train_data, train_label = cifar100_train_dataset.next()
        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = train_data.to(device), train_label.to(device)

        train_pred1 = VGG16(train_data)

        # reset optimizer with zero gradient
        optimizer.zero_grad()

        # choose level 2 hierarchy, 20-class classification
        train_loss1 = VGG16.model_fit(train_pred1, train_label[:, 2], num_output=20)
        train_loss = torch.mean(train_loss1)

        # compute training loss and apply one gradient update
        train_loss.backward()
        optimizer.step()

        # calculate training loss and accuracy
        train_predict_label1 = train_pred1.data.max(1)[1]
        train_acc1 = train_predict_label1.eq(train_label[:, 2]).sum().item() / batch_size

        cost[0] = torch.mean(train_loss1).item()
        cost[1] = train_acc1
        k = k + 1
        avg_cost[index][0:2] += cost / train_batch

    # evaluating test data
    VGG16.eval()
    with torch.no_grad():
        cifar100_test_dataset = iter(cifar100_test_loader)
        for i in range(test_batch):
            test_data, test_label = cifar100_test_dataset.next()
            test_label = test_label.type(torch.LongTensor)
            test_data, test_label = test_data.to(device), test_label.to(device)
            test_pred1 = VGG16(test_data)

            # evaluate on test data
            test_loss1 = VGG16.model_fit(test_pred1, test_label[:, 2], num_output=20)

            # calculate testing loss and accuracy
            test_predict_label1 = test_pred1.data.max(1)[1]
            test_acc1 = test_predict_label1.eq(test_label[:, 2]).sum().item() / batch_size

            cost[0] = torch.mean(test_loss1).item()
            cost[1] = test_acc1
            avg_cost[index][2:] += cost / test_batch

    scheduler.step()
    print('EPOCH: {:04d} ITER: {:04d} | TRAIN [LOSS|ACC.]: {:.4f} {:.4f} || TEST [LOSS|ACC.]: {:.4f} {:.4f}'
          .format(index, k, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3]))
