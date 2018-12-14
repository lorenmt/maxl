import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import glob

from collections import OrderedDict
from torch.autograd import Variable
from create_dataset import *
import re
import matplotlib.pyplot as plt


class LabelGenerator(nn.Module):
    def __init__(self, class_nb):
        super(LabelGenerator, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = class_nb
        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0], filter[0]], bottle_neck=True)])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0], filter[0]], bottle_neck=True)])

        for i in range(4):
            if i < 2:
                self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1], filter[i + 1]], bottle_neck=True))
                self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i], filter[i]], bottle_neck=True))
            else:
                self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1], filter[i + 1]], bottle_neck=False))
                self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i], filter[i]], bottle_neck=False))

        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.pred_task1 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                        nn.Conv2d(in_channels=filter[0], out_channels=1, kernel_size=1, padding=0))

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

    def conv_layer(self, channel, bottle_neck):
        if bottle_neck:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[2]),
                nn.ReLU(inplace=True),
            )

        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[2]),
                nn.ReLU(inplace=True),
            )
        return conv_block

    def mask_softmax(self, x, mask, dim=1):
        mask = mask.type(torch.float)
        logits = torch.exp(x) * mask / torch.sum(torch.exp(x) * mask, dim=dim, keepdim=True)
        return logits

    def forward(self, x, y):
        encoder_conv, decoder_conv, encoder_samp, decoder_samp, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            if i == 0:
                encoder_conv[i] = self.encoder_block[i](x)
                encoder_samp[i], indices[i] = self.down_sampling(encoder_conv[i])
            else:
                encoder_conv[i] = self.encoder_block[i](encoder_samp[i - 1])
                encoder_samp[i], indices[i] = self.down_sampling(encoder_conv[i])

        for i in range(5):
            if i == 0:
                decoder_samp[i] = self.up_sampling(encoder_samp[-1], indices[-i - 1])
                decoder_conv[i] = self.decoder_block[-i - 1](decoder_samp[i])
            else:
                decoder_samp[i] = self.up_sampling(decoder_conv[i - 1], indices[-i - 1])
                decoder_conv[i] = self.decoder_block[-i - 1](decoder_samp[i])

        # build a binary mask
        index = np.zeros([len(self.class_nb), np.sum(self.class_nb)]) + 1e-24
        for i in range(len(self.class_nb)):
            index[i, int(np.sum(self.class_nb[:i])):np.sum(self.class_nb[:i+1])] = 1
        mask = torch.from_numpy(np.moveaxis(index[y].astype(int), -1, 1)).to(device)

        predict = self.pred_task1(decoder_conv[-1])
        label_pred = self.mask_softmax(predict, mask, dim=1)

        return label_pred


class SegNet(nn.Module):
    def __init__(self, class_nb):
        super(SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = class_nb

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0], filter[0]], bottle_neck=True)])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0], filter[0]], bottle_neck=True)])

        for i in range(4):
            if i == 0:
                self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1], filter[i + 1]], bottle_neck=True))
                self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i], filter[i]], bottle_neck=True))
            else:
                self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1], filter[i + 1]], bottle_neck=False))
                self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i], filter[i]], bottle_neck=False))

        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.pred_task1 = nn.Sequential(
            nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=filter[0], out_channels=len(self.class_nb), kernel_size=1, padding=0))
        self.pred_task2 = nn.Sequential(
            nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=filter[0], out_channels=int(np.sum(self.class_nb)), kernel_size=1, padding=0))

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

    def conv_layer(self, channel, bottle_neck):
        if bottle_neck:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[2]),
                nn.ReLU(inplace=True),
            )

        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[2]),
                nn.ReLU(inplace=True),
            )
        return conv_block

    def conv_layer_ff(self, input, weights, type, index):
        if index < 2:
            net = F.conv2d(input, weights['{:s}_block.{:d}.0.weight'.format(type, index)], weights['{:s}_block.{:d}.0.bias'.format(type, index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['{:s}_block.{:d}.1.weight'.format(type, index)], weights['{:s}_block.{:d}.1.bias'.format(type, index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['{:s}_block.{:d}.3.weight'.format(type, index)], weights['{:s}_block.{:d}.3.bias'.format(type, index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['{:s}_block.{:d}.4.weight'.format(type, index)], weights['{:s}_block.{:d}.4.bias'.format(type, index)],
                               training=True)
            net = F.relu(net, inplace=True)
        else:
            net = F.conv2d(input, weights['{:s}_block.{:d}.0.weight'.format(type, index)], weights['{:s}_block.{:d}.0.bias'.format(type, index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['{:s}_block.{:d}.1.weight'.format(type, index)], weights['{:s}_block.{:d}.1.bias'.format(type, index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['{:s}_block.{:d}.3.weight'.format(type, index)], weights['{:s}_block.{:d}.3.bias'.format(type, index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['{:s}_block.{:d}.4.weight'.format(type, index)], weights['{:s}_block.{:d}.4.bias'.format(type, index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['{:s}_block.{:d}.6.weight'.format(type, index)], weights['{:s}_block.{:d}.6.bias'.format(type, index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['{:s}_block.{:d}.7.weight'.format(type, index)], weights['{:s}_block.{:d}.7.bias'.format(type, index)],
                               training=True)
            net = F.relu(net, inplace=True)
        return net

    def pred_layer_ff(self, input, weights, index):
        net = F.conv2d(input, weights['pred_task{:d}.0.weight'.format(index)], weights['pred_task{:d}.0.bias'.format(index)], padding=1)
        net = F.conv2d(net, weights['pred_task{:d}.1.weight'.format(index)], weights['pred_task{:d}.1.bias'.format(index)], padding=0)
        return net

    def forward(self, x, weights=None):
        encoder_conv, decoder_conv, encoder_samp, decoder_samp, indices = ([0] * 5 for _ in range(5))

        if weights is None:
            for i in range(5):
                if i == 0:
                    encoder_conv[i] = self.encoder_block[i](x)
                    encoder_samp[i], indices[i] = self.down_sampling(encoder_conv[i])
                else:
                    encoder_conv[i] = self.encoder_block[i](encoder_samp[i - 1])
                    encoder_samp[i], indices[i] = self.down_sampling(encoder_conv[i])

            for i in range(5):
                if i == 0:
                    decoder_samp[i] = self.up_sampling(encoder_samp[-1], indices[-1])
                    decoder_conv[i] = self.decoder_block[-i - 1](decoder_samp[i])
                else:
                    decoder_samp[i] = self.up_sampling(decoder_conv[i - 1], indices[-i - 1])
                    decoder_conv[i] = self.decoder_block[-i - 1](decoder_samp[i])

            t1_pred = F.log_softmax(self.pred_task1(decoder_conv[-1]), dim=1)
            t2_pred = F.softmax(self.pred_task2(decoder_conv[-1]), dim=1)

        else:
            for i in range(5):
                if i == 0:
                    encoder_conv[i] = self.conv_layer_ff(x, weights, type='encoder', index=i)
                    encoder_samp[i], indices[i] = self.down_sampling(encoder_conv[i])
                else:
                    encoder_conv[i] = self.conv_layer_ff(encoder_samp[i - 1], weights, type='encoder', index=i)
                    encoder_samp[i], indices[i] = self.down_sampling(encoder_conv[i])

            for i in range(5):
                if i == 0:
                    decoder_samp[i] = self.up_sampling(encoder_samp[-1], indices[-1])
                    decoder_conv[i] = self.conv_layer_ff(decoder_samp[i], weights, type='decoder', index=4 - i)
                else:
                    decoder_samp[i] = self.up_sampling(decoder_conv[i - 1], indices[-i - 1])
                    decoder_conv[i] = self.conv_layer_ff(decoder_samp[i], weights, type='decoder', index=4 - i)

            t1_pred = F.log_softmax(self.pred_layer_ff(decoder_conv[-1], weights, index=1), dim=1)
            t2_pred = F.softmax(self.pred_layer_ff(decoder_conv[-1], weights, index=2), dim=1)

        return t1_pred, t2_pred

    def model_fit(self, x_pred, x_output, pri=True):
        if not pri:
            binary_mask = (x_output != -1).type(torch.FloatTensor).unsqueeze(1).to(device)
            loss = torch.mean(torch.sum(-x_output * torch.log(x_pred + 1e-24) * binary_mask, dim=1))
        else:
            loss = F.nll_loss(x_pred, x_output, ignore_index=-1, reduce=True)
        return loss

    def compute_miou(self, x_pred, x_output, batch_size):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        for i in range(batch_size):
            true_class = 0
            first_switch = True
            for j in range(len(self.class_nb)):
                pred_mask = torch.eq(x_pred_label[i], Variable(j*torch.ones(x_pred_label[i].shape).type(torch.LongTensor).cuda()))
                true_mask = torch.eq(x_output_label[i], Variable(j*torch.ones(x_output_label[i].shape).type(torch.LongTensor).cuda()))
                mask_comb = pred_mask + true_mask
                union     = torch.sum((mask_comb > 0).type(torch.FloatTensor))
                intsec    = torch.sum((mask_comb > 1).type(torch.FloatTensor))
                if union.data.numpy() == 0:
                    continue
                if first_switch:
                    class_prob = intsec / union
                    first_switch = False
                else:
                    class_prob = intsec / union + class_prob
                true_class += 1
            if i == 0:
                batch_avg = class_prob / true_class
            else:
                batch_avg = class_prob / true_class + batch_avg
        return batch_avg / batch_size

    def compute_iou(self, x_pred, x_output, batch_size):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        for i in range(batch_size):
            if i == 0:
                pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                            torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
            else:
                pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                            torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
        return pixel_acc / batch_size

    def model_entropy(self, x_pred1):
        x_pred1 = torch.mean(x_pred1, dim=0)
        loss1 = x_pred1 * torch.log(x_pred1 + 1e-24)
        return torch.sum(loss1)


# define model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LabelGenerator = LabelGenerator([5]*13).to(device)
gen_optimizer = optim.SGD(LabelGenerator.parameters(), lr=1e-3, weight_decay=5e-3)
gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=50, gamma=0.5)


SegNet = SegNet([5]*13).to(device)
optimizer = optim.SGD(SegNet.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# define dataset path
trans_train = transforms.Compose([
    transforms.ToTensor(),
])
trans_test = transforms.Compose([
    transforms.ToTensor(),
])

nyu_train = NYUv2(data_path='../multi-task-attention/nyuv2', type='train', task='semantic', transform=trans_train)
nyu_test = NYUv2(data_path='../multi-task-attention/nyuv2', type='val', task='semantic', transform=trans_test)

batch_size = 4
kwargs = {'num_workers': 1, 'pin_memory': True}
nyu_train_loader = torch.utils.data.DataLoader(
    dataset=nyu_train,
    batch_size=batch_size,
    shuffle=True)

nyu_val_loader = torch.utils.data.DataLoader(
    dataset=nyu_test,
    batch_size=batch_size,
    shuffle=True)


# load model
# SegNet.load_state_dict(torch.load('model_weights/segnet_semantic_city_19'))

# define parameters
total_epoch = 200
train_batch = len(nyu_train_loader)-1
val_batch   = len(nyu_val_loader)-1
k = 0
avg_cost = np.zeros([total_epoch, 11], dtype=np.float32)
vgg_lr = 0.01
for epoch in range(total_epoch):
    index = epoch
    cost = np.zeros(4, dtype=np.float32)
    if (epoch + 1) % 50 == 0:
        vgg_lr = vgg_lr * 0.5
    scheduler.step()
    gen_scheduler.step()

    # iteration for all batches
    nyu_train_dataset = iter(nyu_train_loader)
    for i in range(train_batch):
        train_data, train_label = nyu_train_dataset.next()
        train_data, train_label = train_data.to(device), train_label.to(device)
        #train_data = train_data.type(torch.)
        train_label = train_label.type(torch.long)

        train_pred1, train_pred2 = SegNet(train_data)
        train_pred3 = LabelGenerator(train_data, train_label.cpu().numpy())

        optimizer.zero_grad()

        train_loss1 = SegNet.model_fit(train_pred1, train_label, pri=True)
        train_loss2 = SegNet.model_fit(train_pred2, train_pred3, pri=False)

        (train_loss1 + train_loss2).backward()
        optimizer.step()

        cost[0] = train_loss1.item()
        cost[1] = SegNet.compute_miou(torch.exp(train_pred1), train_label, batch_size).item()
        cost[2] = SegNet.compute_iou(torch.exp(train_pred1), train_label, batch_size).item()
        cost[3] = train_loss2.item()
        avg_cost[index][0:4] += cost[0:4] / train_batch

    nyu_train_dataset = iter(nyu_train_loader)
    #nyu_train_dataset2 = iter(nyu_train_loader)
    for i in range(train_batch):
        train_data, train_label = nyu_train_dataset.next()
        train_data, train_label = train_data.to(device), train_label.to(device)
        train_label = train_label.type(torch.long)

        train_pred1, train_pred2 = SegNet(train_data)
        train_pred3 = LabelGenerator(train_data, train_label.cpu().numpy())

        gen_optimizer.zero_grad()

        train_loss1 = SegNet.model_fit(train_pred1, train_label, pri=True)
        train_loss2 = SegNet.model_fit(train_pred2, train_pred3, pri=False)
        train_loss3 = SegNet.model_entropy(train_pred3)
        train_loss = torch.mean(train_loss1) + torch.mean(train_loss2)

        cost[0] = torch.mean(train_loss1).item()
        cost[1] = SegNet.compute_miou(torch.exp(train_pred1), train_label, batch_size).item()

        fast_weights = OrderedDict((name, param) for (name, param) in SegNet.named_parameters())
        grads = torch.autograd.grad(train_loss, SegNet.parameters(), retain_graph=True)
        data = [p.data for p in list(SegNet.parameters())]

        fast_weights = OrderedDict((name, param - vgg_lr * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))

        #train_data, train_label = nyu_train_dataset2.next()
        #train_data, train_label = train_data.to(device), train_label.to(device)
        #train_label = train_label.type(torch.long)
        train_pred1, train_pred2 = SegNet(train_data, fast_weights)
        train_loss1 = SegNet.model_fit(train_pred1, train_label, pri=True)

        (torch.mean(train_pred1) +0.1*torch.mean(train_loss3)).backward()
        gen_optimizer.step()

        cost[2] = torch.mean(train_loss1).item()
        cost[3] = SegNet.compute_miou(torch.exp(train_pred1), train_label, batch_size).item()
        avg_cost[index][4:8] += cost[0:4] / train_batch
        k = k + 1

    # evaluating test data
    with torch.no_grad():  # operations inside don't track history
        nyu_val_dataset = iter(nyu_val_loader)
        for i in range(val_batch):
            test_data, test_label = nyu_val_dataset.next()
            test_data, test_label = test_data.to(device), test_label.to(device)
            test_label = test_label.type(torch.long)

            test_pred1, test_pred2 = SegNet(test_data)
            test_loss = SegNet.model_fit(test_pred1, test_label)

            cost[0] = torch.mean(test_loss).item()
            cost[1] = SegNet.compute_miou(torch.exp(test_pred1), test_label, batch_size).item()
            cost[2] = SegNet.compute_iou(torch.exp(test_pred1), test_label, batch_size).item()

            avg_cost[index][8:11] += cost[0:3] / val_batch

    # print('Epoch: {:04d} Iteration: {:04d}| LABEL-TRAIN: {:.4f} | LABEL-TEST: {:.4f} '
    #           .format(index, k, avg_cost[index, 0], avg_cost[index, 3]))
    print('Epoch: {:04d} Iteration: {:04d}| TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f}  Val: {:.4f} {:.4f} | {:.4f} {:.4f} Test: {:.4f} {:.4f} {:.4f}'
              .format(index, k, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3], avg_cost[index, 4], avg_cost[index, 5],
                      avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                      avg_cost[index, 10]))


# torch.save(SegNet.state_dict(), 'model_weights/segnet_semantic_nyu')
# np.save('cost/cost_segnet_nyu_semantic.npy', avg_cost)

