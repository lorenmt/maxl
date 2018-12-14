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
    def __init__(self):
        super(VGG16, self).__init__()
        filter = [64, 128, 256, 512, 512]

        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        self.classifier = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], 20),
        )

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
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)
        # loss = x_output_onehot * torch.log(x_pred + 1e-20) # normal cross entropy
        loss = x_output_onehot * (1 - x_pred) ** 2 * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)


# load CIFAR100 dataset
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


cifar100_train_set = CIFAR100(data_path='dataset', train=True, transform=trans_train)
cifar100_test_set = CIFAR100(data_path='dataset', train=False, transform=trans_test)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

batch_size = 100
kwargs = {'num_workers': 1, 'pin_memory': True}
cifar100_train_loader = torch.utils.data.DataLoader(
    dataset=cifar100_train_set,
    batch_size=batch_size,
    sampler=sampler.RandomSampler(np.arange(0, 50000)))

cifar100_test_loader = torch.utils.data.DataLoader(
    dataset=cifar100_test_set,
    batch_size=batch_size,
    shuffle=True)


# define model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VGG16 = VGG16().to(device)
optimizer = optim.SGD(VGG16.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# define parameters
total_epoch = 200
train_batch = len(cifar100_train_loader)
test_batch = len(cifar100_test_loader)
k = 0
avg_cost = np.zeros([total_epoch, 8], dtype=np.float32)
for epoch in range(total_epoch):
    index = epoch
    cost = np.zeros(4, dtype=np.float32)
    scheduler.step()

    # iteration for all batches
    cifar100_train_dataset = iter(cifar100_train_loader)
    for i in range(train_batch):
        # evaluating training datata
        train_data, train_label, _ = cifar100_train_dataset.next()
        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = train_data.to(device), train_label.to(device)
        train_pred1 = VGG16(train_data)
        optimizer.zero_grad()
        train_loss1 = VGG16.model_fit(train_pred1, train_label[:, 2], 20)
        train_loss = torch.mean(train_loss1)

        train_loss.backward()
        optimizer.step()

        train_predict_label1 = train_pred1.data.max(1)[1]

        train_acc1 = train_predict_label1.eq(train_label[:, 2]).sum().item() / batch_size

        cost[0] = torch.mean(train_loss1).item()
        cost[1] = train_acc1
        k = k + 1
        avg_cost[index][0:4] += cost / train_batch

    # evaluating test data
    with torch.no_grad():
        cifar100_test_dataset = iter(cifar100_test_loader)
        for i in range(test_batch):
            test_data, test_label, _ = cifar100_test_dataset.next()
            test_label = test_label.type(torch.LongTensor)
            test_data, test_label = test_data.to(device), test_label.to(device)
            test_pred1 = VGG16(test_data)
            test_loss1 = VGG16.model_fit(test_pred1, test_label[:, 2], 20)

            test_predict_label1 = test_pred1.data.max(1)[1]

            test_acc1 = test_predict_label1.eq(test_label[:, 2]).sum().item() / batch_size
            cost[0] = torch.mean(test_loss1).item()
            cost[1] = test_acc1

            avg_cost[index][4:] += cost / test_batch

    print('Epoch: {:04d} Iteration: {:04d} | CIFAR100-TRAIN: {:.4f} {:.4f}  || '
          'CIFAR100-TEST: {:.4f} {:.4f} '
          .format(epoch, k, avg_cost[index][0], avg_cost[index][1],
                  avg_cost[index][4], avg_cost[index][5]))

#torch.save(VGG16.state_dict(), 'model_weights/cifar10_human_old')
#np.save('loss/cifar10_human_old.npy', avg_cost)
