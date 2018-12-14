import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from create_dataset import CIFAR100
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        filter = [64, 128, 256, 512, 512]

        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        self.classifier1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 100),
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

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

        t1_pred = self.classifier1(g_block5.view(g_block5.size(0), -1))
        t1_pred = F.log_softmax(t1_pred, dim=1)

        t2_pred = self.classifier2(g_block5.view(g_block5.size(0), -1))
        t2_pred = F.log_softmax(t2_pred, dim=1)

        return t1_pred, t2_pred

    def model_fit(self, x_pred1, x_output1, x_pred2, x_output2):
        loss1 = F.nll_loss(x_pred1, x_output1)
        loss2 = F.nll_loss(x_pred2, x_output2)
        return loss1, loss2


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

batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True}
cifar100_train_loader = torch.utils.data.DataLoader(
    dataset=cifar100_train_set,
    batch_size=batch_size,
    shuffle=True, **kwargs)
cifar100_test_loader = torch.utils.data.DataLoader(
    dataset=cifar100_test_set,
    batch_size=batch_size,
    shuffle=True, **kwargs)


# define model
VGG16 = VGG16().cuda()
optimizer = optim.SGD(VGG16.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-3, nesterov=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

#VGG16.load_state_dict(torch.load('model_weights/vgg_2'))

# define parameters
total_epoch = 300
train_batch = len(cifar100_train_loader) - 1
test_batch = len(cifar100_test_loader) - 1
k = 0
avg_cost = np.zeros([total_epoch, 8], dtype=np.float32)
lambda_i = np.zeros(2)
switch = True
for epoch in range(total_epoch):
    index = epoch
    cost = np.zeros(4, dtype=np.float32)
    # apply Dynamic Weight Average
    if switch:
        if index < 300:
            lambda_i[0] = 0.5
            lambda_i[1] = 0.5
        else:
            optimizer = optim.SGD(VGG16.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-3, nesterov=True)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
            lambda_i[0] = 1
            lambda_i[1] = 0
            switch = False

    scheduler.step()
    # iteration for all batches
    cifar100_train_dataset = iter(cifar100_train_loader)
    for i in range(train_batch):
        # evaluating training datata
        train_data, train_label = cifar100_train_dataset.next()

        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = Variable(train_data.cuda()), Variable(train_label.cuda())

        train_pred1, train_pred2 = VGG16(train_data)

        optimizer.zero_grad()
        train_loss1, train_loss2 = VGG16.model_fit(train_pred1, train_label[:,1], train_pred2, train_label[:,3])
        train_loss = float(lambda_i[0])*torch.mean(train_loss1) + float(lambda_i[1])*torch.mean(train_loss2)
        train_loss.backward()
        # nn.utils.clip_grad_norm(MTAN_VGG.parameters(), 1)
        optimizer.step()

        train_predict_label1 = train_pred1.data.max(1)[1]
        train_predict_label2 = train_pred2.data.max(1)[1]

        train_acc1 = train_predict_label1.eq(train_label[:,1].contiguous().data).sum() / batch_size
        train_acc2 = train_predict_label2.eq(train_label[:,3].contiguous().data).sum() / batch_size

        cost[0] = torch.mean(train_loss1).data[0]
        cost[1] = train_acc1
        cost[2] = torch.mean(train_loss2).data[0]
        cost[3] = train_acc2
        k = k + 1
        avg_cost[index][0:4] += cost / train_batch


        # evaluating test data
    cifar100_test_dataset = iter(cifar100_test_loader)
    for i in range(test_batch):
        test_data, test_label = cifar100_test_dataset.next()
        test_label = test_label.type(torch.LongTensor)

        test_data, test_label = Variable(test_data.cuda(), volatile=True), Variable(test_label.cuda(), volatile=True)
        test_pred1, test_pred2 = VGG16(test_data)
        test_loss1, test_loss2 = VGG16.model_fit(test_pred1, test_label[:,1], test_pred2, test_label[:,3])

        test_predict_label1 = test_pred1.data.max(1)[1]
        test_predict_label2 = test_pred2.data.max(1)[1]

        test_acc1 = test_predict_label1.eq(test_label[:,1].contiguous().data).sum() / batch_size
        test_acc2 = test_predict_label2.eq(test_label[:,3].contiguous().data).sum() / batch_size

        cost[0] = torch.mean(test_loss1).data[0]
        cost[1] = test_acc1
        cost[2] = torch.mean(test_loss2).data[0]
        cost[3] = test_acc2

        avg_cost[index][4:] += cost / test_batch

    print('Epoch: {:04d} Iteration: {:04d} | CIFAR100-TRAIN: {:.4f} {:.4f} | {:.4f} {:.4f} | CIFAR100-TEST: {:.4f} {:.4f} | {:.4f} {:.4f}'
          .format(epoch, k, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3],
                  avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7]))


torch.save(VGG16.state_dict(), 'model_weights/cifar100_vgg_10_100')
np.save('loss/cifar100_vgg_10_100.npy', avg_cost)

