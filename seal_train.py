import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from create_dataset import *
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler
import matplotlib.pyplot as plt
from sklearn import manifold



class LabelGenerator(nn.Module):
    """
    Meta-generator class.
    """
    def __init__(self, class_nb):
        super(LabelGenerator, self).__init__()
        filter = [64, 128, 256, 512, 512]
        self.class_nb = class_nb

        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        self.classifier = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], int(np.sum(self.class_nb))),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
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

    def mask_softmax(self, x, mask, dim=1):
        logits = torch.exp(x) * mask / torch.sum(torch.exp(x) * mask, dim=dim, keepdim=True).repeat(1, np.sum(self.class_nb))
        return logits

    def forward(self, x, y):
        g_block1 = self.block1(x)
        g_block2 = self.block2(g_block1)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        g_block5 = self.block5(g_block4)

        # build a binary mask
        index = torch.zeros([len(self.class_nb), np.sum(self.class_nb)]) + 1e-8
        for i in range(len(self.class_nb)):
            index[i, int(np.sum(self.class_nb[:i])):np.sum(self.class_nb[:i+1])] = 1
        mask = index[y].to(device)

        predict = self.classifier(g_block5.view(g_block5.size(0), -1))
        label_pred = self.mask_softmax(predict, mask, dim=1)

        return label_pred


class VGG16(nn.Module):
    """
    Multi-task evaluator class.
    """
    def __init__(self, class_nb):
        super(VGG16, self).__init__()
        filter = [64, 128, 256, 512, 512]

        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        self.classifier1 = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], len(class_nb)),
            nn.Softmax(dim=1)
        )

        self.classifier2 = nn.Sequential(
            # TODO weird -> remove these layers
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], int(np.sum(class_nb))),
            nn.Softmax(dim=1)
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


    # TODO why is this necessary?
    def conv_layer_ff(self, input, weights, index):
        if index < 3:
            net = F.conv2d(input, weights['block{:d}.0.weight'.format(index)], weights['block{:d}.0.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.1.weight'.format(index)], weights['block{:d}.1.bias'.format(index)], training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['block{:d}.3.weight'.format(index)], weights['block{:d}.3.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.4.weight'.format(index)], weights['block{:d}.4.bias'.format(index)], training=True)
            net = F.relu(net, inplace=True)
            net = F.max_pool2d(net, kernel_size=2, stride=2, )
        else:
            net = F.conv2d(input, weights['block{:d}.0.weight'.format(index)], weights['block{:d}.0.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.1.weight'.format(index)], weights['block{:d}.1.bias'.format(index)], training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['block{:d}.3.weight'.format(index)], weights['block{:d}.3.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.4.weight'.format(index)], weights['block{:d}.4.bias'.format(index)], training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['block{:d}.6.weight'.format(index)], weights['block{:d}.6.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.7.weight'.format(index)], weights['block{:d}.7.bias'.format(index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.max_pool2d(net, kernel_size=2, stride=2)

        return net


    # TODO why is this necessary?
    def dense_layer_ff(self, input, weights, index):
        net = F.linear(input, weights['classifier{:d}.0.weight'.format(index)], weights['classifier{:d}.0.bias'.format(index)])
        net = F.relu(net, inplace=True)
        net = F.linear(net, weights['classifier{:d}.2.weight'.format(index)], weights['classifier{:d}.2.bias'.format(index)])
        net = F.softmax(net, dim=1)
        return net


    def forward(self, x, weights=None):
        if weights is None:
            g_block1 = self.block1(x)
            g_block2 = self.block2(g_block1)
            g_block3 = self.block3(g_block2)
            g_block4 = self.block4(g_block3)
            g_block5 = self.block5(g_block4)

            t1_pred = self.classifier1(g_block5.view(g_block5.size(0), -1))
            t2_pred = self.classifier2(g_block5.view(g_block5.size(0), -1))

        else:
            g_block1 = self.conv_layer_ff(x, weights, 1)
            g_block2 = self.conv_layer_ff(g_block1, weights, 2)
            g_block3 = self.conv_layer_ff(g_block2, weights, 3)
            g_block4 = self.conv_layer_ff(g_block3, weights, 4)
            g_block5 = self.conv_layer_ff(g_block4, weights, 5)

            t1_pred = self.dense_layer_ff(g_block5.view(g_block5.size(0), -1), weights, 1)
            t2_pred = self.dense_layer_ff(g_block5.view(g_block5.size(0), -1), weights, 2)

        return t1_pred, t2_pred


    def model_fit(self, x_pred, x_output, pri=True, num_output=3):
        if not pri:
            x_output_onehot = x_output
        else:
            x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
            x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)
        # loss = x_output_onehot * torch.log(x_pred + 1e-24) # normal cross_entropy
        loss = x_output_onehot * (1 - x_pred)**2 * torch.log(x_pred + 1e-24)
        return torch.sum(-loss, dim=1)


    def model_entropy(self, x_pred1):
        x_pred1 = torch.mean(x_pred1, dim=0)
        loss1 = x_pred1 * torch.log(x_pred1 + 1e-24)
        loss1 = torch.sum(loss1)
        return loss1


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

data_path = '/Users/FabianFalck/Documents/[03]SelfAuxLearning/Data'

cifar100_train_set = CIFAR100(data_path=data_path, train=True, transform=trans_train, auxiliary=None)
cifar100_test_set = CIFAR100(data_path=data_path, train=False, transform=trans_test)

batch_size = 100
kwargs = {'num_workers': 1, 'pin_memory': True}
cifar100_train_loader = torch.utils.data.DataLoader(
    dataset=cifar100_train_set,
    batch_size=batch_size,
    sampler=sampler.RandomSampler(np.arange(0, 50000)))
    #shuffle=True)

cifar100_val_loader = torch.utils.data.DataLoader(
    dataset=cifar100_train_set,
    batch_size=batch_size,
    sampler=sampler.RandomSampler(np.arange(0, 50000)))
    #shuffle=True)

cifar100_test_loader = torch.utils.data.DataLoader(
    dataset=cifar100_test_set,
    batch_size=batch_size,
    shuffle=True)

# define model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LabelGenerator = LabelGenerator(class_nb=[5]*20).to(device)
gen_optimizer = optim.SGD(LabelGenerator.parameters(), lr=1e-3, weight_decay=5e-4)
gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=50, gamma=0.5)

# define parameters
total_epoch = 200
train_batch = len(cifar100_train_loader)
val_batch = len(cifar100_val_loader)
test_batch = len(cifar100_test_loader)

VGG16_model = VGG16(class_nb=[5]*20).to(device)
optimizer = optim.SGD(VGG16_model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
avg_cost = np.zeros([total_epoch, 10], dtype=np.float32)
vgg_lr = 0.01
for epoch in range(total_epoch):
    index = epoch
    cost = np.zeros(4, dtype=np.float32)

    if (epoch + 1) % 50 == 0:
       vgg_lr = vgg_lr * 0.5
    scheduler.step()
    gen_scheduler.step()

    # iteration for all batches
    cifar100_train_dataset = iter(cifar100_train_loader)
    for i in range(train_batch):
        print("batch %d"%i)

        # evaluating training datata
        train_data, train_label, _ = cifar100_train_dataset.next()
        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = train_data.to(device), train_label.to(device)
        train_pred1, train_pred2 = VGG16_model(train_data)
        train_pred3 = LabelGenerator(train_data, train_label[:, 2])
        # train_label[:, i], i= 0,1,2,3 represents 3, 10, 20, 100-class

        optimizer.zero_grad()
        gen_optimizer.zero_grad()

        train_loss1 = VGG16_model.model_fit(train_pred1, train_label[:, 2], pri=True, num_output=20)
        train_loss2 = VGG16_model.model_fit(train_pred2, train_pred3, pri=False, num_output=100)
        train_loss3 = VGG16_model.model_entropy(train_pred3)

        train_loss = torch.mean(train_loss1) + torch.mean(train_loss2)
        train_loss.backward()

        optimizer.step()

        train_predict_label1 = train_pred1.data.max(1)[1]
        train_predict_label2 = train_pred2.data.max(1)[1]
        train_acc1 = train_predict_label1.eq(train_label[:, 2]).sum().item() / batch_size
        train_acc2 = train_predict_label2.eq(train_pred3.data.max(1)[1]).sum().item() / batch_size

        cost[0] = torch.mean(train_loss1).item()
        cost[1] = train_acc1
        cost[2] = torch.mean(train_loss2).item()
        cost[3] = train_acc2
        avg_cost[index][0:4] += cost / train_batch

    # evaluating meta data
    cifar100_val_dataset = iter(cifar100_val_loader)
    for i in range(val_batch):
        test_data, test_label, _ = cifar100_val_dataset.next()
        test_label = test_label.type(torch.LongTensor)
        test_data, test_label = test_data.to(device), test_label.to(device)
        test_pred1, test_pred2 = VGG16_model(test_data)
        test_pred3 = LabelGenerator(test_data, test_label[:, 2])

        test_loss1 = VGG16_model.model_fit(test_pred1, test_label[:, 2], pri=True, num_output=20)
        test_loss2 = VGG16_model.model_fit(test_pred2, test_pred3, pri=False, num_output=100)
        test_loss3 = VGG16_model.model_entropy(test_pred3)
        test_loss = torch.mean(test_loss1) + torch.mean(test_loss2)

        test_predict_label1 = test_pred1.data.max(1)[1]
        test_acc1 = test_predict_label1.eq(test_label[:, 2]).sum().item() / batch_size
        cost[0] = torch.mean(test_loss1).item()
        cost[1] = test_acc1

        # ----  # FROM MAMAL 2ND ORDER DERIVATIVE TRICK

        fast_weights = OrderedDict((name, param) for (name, param) in VGG16_model.named_parameters())

        grads = torch.autograd.grad(test_loss, VGG16_model.parameters(), retain_graph=True)
        data = [p.data for p in list(VGG16_model.parameters())]

        fast_weights = OrderedDict((name, param - vgg_lr * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))

        # ----

        test_pred1, test_pred2 = VGG16_model.forward(test_data, fast_weights)
        test_loss1 = VGG16_model.model_fit(test_pred1, test_label[:, 2], pri=True, num_output=20)

        (torch.mean(test_loss1) + 0.2 * torch.mean(test_loss3)).backward()
        gen_optimizer.step()

        test_predict_label1 = test_pred1.data.max(1)[1]
        test_acc1 = test_predict_label1.eq(test_label[:, 2]).sum().item() / batch_size

        cost[2] = torch.mean(test_loss1).item()
        cost[3] = test_acc1
        avg_cost[index][4:8] += cost[0:4] / (val_batch)

    with torch.no_grad():
        cifar100_test_dataset = iter(cifar100_test_loader)
        for i in range(test_batch):
            test_data, test_label, _ = cifar100_test_dataset.next()
            test_label = test_label.type(torch.LongTensor)
            test_data, test_label = test_data.to(device), test_label.to(device)
            test_pred1, test_pred2 = VGG16_model(test_data)

            test_loss1 = VGG16_model.model_fit(test_pred1, test_label[:, 2], pri=True, num_output=20)

            test_predict_label1 = test_pred1.data.max(1)[1]
            test_acc1 = test_predict_label1.eq(test_label[:, 2]).sum().item() / batch_size

            cost[0] = torch.mean(test_loss1).item()
            cost[1] = test_acc1

            avg_cost[index][8:] += cost[0:2] / test_batch

    print('Iter {:04d} | CIFAR100-TRAIN: {:.4f} {:.4f} | {:.4f} {:.4f} || CIFAR100-VAL: {:.4f} {:.4f} {:.4f} {:.4f} || TEST: {:.4f} {:.4f}'
          .format(epoch, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3],
                  avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7], avg_cost[index][8], avg_cost[index][9]))
