from collections import OrderedDict
from create_dataset import *

import numpy as np
import copy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler


# --------------------------------------------------------------------------------
# Define network
# --------------------------------------------------------------------------------
class LabelGenerator(nn.Module):
    def __init__(self, psi):
        super(LabelGenerator, self).__init__()
        """
            label-generation network:
            takes the input and generates auxiliary labels with masked softmax for an auxiliary task.
        """
        filter = [64, 128, 256, 512, 512]
        self.class_nb = psi

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        # define fc-layers in VGG-16 (output auxiliary classes \sum_i\psi[i])
        self.classifier = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], int(np.sum(self.class_nb))),
        )

        # apply weight initialisation
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

    # define masked softmax
    def mask_softmax(self, x, mask, dim=1):
        logits = torch.exp(x) * mask / torch.sum(torch.exp(x) * mask, dim=dim, keepdim=True)
        return logits

    def forward(self, x, y):
        g_block1 = self.block1(x)
        g_block2 = self.block2(g_block1)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        g_block5 = self.block5(g_block4)

        # build a binary mask by psi, we add epsilon=1e-8 to avoid nans
        index = torch.zeros([len(self.class_nb), np.sum(self.class_nb)]) + 1e-8
        for i in range(len(self.class_nb)):
            index[i, int(np.sum(self.class_nb[:i])):np.sum(self.class_nb[:i+1])] = 1
        mask = index[y].to(device)

        predict = self.classifier(g_block5.view(g_block5.size(0), -1))
        label_pred = self.mask_softmax(predict, mask, dim=1)

        return label_pred


class VGG16(nn.Module):
    def __init__(self, psi):
        super(VGG16, self).__init__()
        """
            multi-task network:
            takes the input and predicts primary and auxiliary labels (same network structure as in human)
        """
        filter = [64, 128, 256, 512, 512]

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        # primary task prediction
        self.classifier1 = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], len(psi)),
            nn.Softmax(dim=1)
        )

        # auxiliary task prediction
        self.classifier2 = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], int(np.sum(psi))),
            nn.Softmax(dim=1)
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

        t1_pred = self.classifier1(g_block5.view(g_block5.size(0), -1))
        t2_pred = self.classifier2(g_block5.view(g_block5.size(0), -1))
        return t1_pred, t2_pred

    def model_fit(self, x_pred, x_output, pri=True, num_output=3):
        if not pri:
            # generated auxiliary label is a soft-assignment vector (no need to change into one-hot vector)
            x_output_onehot = x_output
        else:
            # convert a single label into a one-hot vector
            x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
            x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # apply focal loss
        loss = x_output_onehot * (1 - x_pred)**2 * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)

    def model_entropy(self, x_pred1):
        # compute entropy loss
        x_pred1 = torch.mean(x_pred1, dim=0)
        loss1 = x_pred1 * torch.log(x_pred1 + 1e-20)
        return torch.sum(loss1)


# --------------------------------------------------------------------------------
# Define MAXL framework with first order approximation
# --------------------------------------------------------------------------------
class MetaAuxiliaryLearning:
    def __init__(self, multi_task_net, label_generator):
        self.multi_task_net = multi_task_net
        self.multi_task_net_ = copy.deepcopy(multi_task_net)
        self.label_generator = label_generator

    def unrolled_backward(self, train_x, train_y, alpha, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        #  compute unrolled multi-task network theta_1^+ (virtual step)
        train_pred1, train_pred2 = self.multi_task_net(train_x)
        train_pred3 = self.label_generator(train_x, train_y)

        train_loss1 = self.multi_task_net.model_fit(train_pred1, train_y, pri=True, num_output=20)
        train_loss2 = self.multi_task_net.model_fit(train_pred2, train_pred3, pri=False, num_output=100)
        train_loss3 = self.multi_task_net.model_entropy(train_pred3)

        loss = torch.mean(train_loss1) + torch.mean(train_loss2)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.multi_task_net.parameters())

        # do virtual step: theta_1^+ = theta_1 - alpha * (primary loss + auxiliary loss)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.multi_task_net.parameters(), self.multi_task_net_.parameters(),
                                             gradients):
                if 'momentum' in model_optim.param_groups[0].keys():  # used in SGD with momentum
                    if model_optim.param_groups[0]['momentum'] == 0:
                        m = 0
                    else:
                        m = model_optim.state[weight].get('momentum_buffer', 0.) * model_optim.param_groups[0]['momentum']
                else:
                    m = 0
                weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]['weight_decay'] * weight))

        # meta-training step: updating theta_2
        train_pred1, train_pred2 = self.multi_task_net_(train_x)
        train_loss1 = self.multi_task_net.model_fit(train_pred1, train_y, pri=True, num_output=20)
        train_loss2 = F.mse_loss(train_pred2, torch.zeros_like(train_pred2, device=train_x.device))  # dummy loss function

        # multi-task loss (set 0 weighting for train_loss 2;
        # so that to make sure the gradient of auxiliary prediction head is not None)
        loss = torch.mean(train_loss1) + 0 * torch.mean(train_loss2) + 0.2 * torch.mean(train_loss3)

        # compute hessian (finite difference approximation)
        model_weights_ = tuple(self.multi_task_net_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_)
        hessian = self.compute_hessian(d_model, train_x, train_y)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip(self.label_generator.parameters(), hessian):
                mw.grad = - alpha * h

    def compute_hessian(self, d_model, train_x, train_y):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # theta_1^l = theta_1 + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.multi_task_net.parameters(), d_model):
                p += eps * d

        train_pred1, train_pred2 = self.multi_task_net(train_x)
        train_pred3 = self.label_generator(train_x, train_y)

        train_loss1 = self.multi_task_net.model_fit(train_pred1, train_y, pri=True, num_output=20)
        train_loss2 = self.multi_task_net.model_fit(train_pred2, train_pred3, pri=False, num_output=100)
        loss = torch.mean(train_loss1) + torch.mean(train_loss2)
        d_weight_p = torch.autograd.grad(loss, self.label_generator.parameters())

        # theta_1^r = theta_1 - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.multi_task_net.parameters(), d_model):
                p -= 2 * eps * d

        train_pred1, train_pred2 = self.multi_task_net(train_x)
        train_pred3 = self.label_generator(train_x, train_y)

        train_loss1 = self.multi_task_net.model_fit(train_pred1, train_y, pri=True, num_output=20)
        train_loss2 = self.multi_task_net.model_fit(train_pred2, train_pred3, pri=False, num_output=100)
        loss = torch.mean(train_loss1) + torch.mean(train_loss2)
        d_weight_n = torch.autograd.grad(loss, self.label_generator.parameters())

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.multi_task_net.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian


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

# define label-generation model,
# and optimiser with learning rate 1e-3, drop half for every 50 epochs, weight_decay=5e-4,
psi = [5]*20  # for each primary class split into 5 auxiliary classes, with total 100 auxiliary classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LabelGenerator = LabelGenerator(psi=psi).to(device)
gen_optimizer = optim.SGD(LabelGenerator.parameters(), lr=1e-3, weight_decay=5e-4)
gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=50, gamma=0.5)

# define parameters
total_epoch = 200
train_batch = len(cifar100_train_loader)
test_batch = len(cifar100_test_loader)

# define multi-task network, and optimiser with learning rate 0.01, drop half for every 50 epochs
VGG16_model = VGG16(psi=psi).to(device)
optimizer = optim.SGD(VGG16_model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# define MAXL framework
maxl = MetaAuxiliaryLearning(multi_task_net=VGG16_model, label_generator=LabelGenerator)

avg_cost = np.zeros([total_epoch, 5], dtype=np.float32)
k = 0
for index in range(total_epoch):
    cost = np.zeros(4, dtype=np.float32)

    # evaluate training data (training-step, update on theta_1)
    VGG16_model.train()
    cifar100_train_dataset = iter(cifar100_train_loader)
    for i in range(train_batch):
        train_data, train_label = cifar100_train_dataset.next()
        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = train_data.to(device), train_label.to(device)

        train_pred1, train_pred2 = VGG16_model(train_data)
        train_pred3 = LabelGenerator(train_data, train_label[:, 2])  # generate auxiliary labels

        # reset optimizers with zero gradient
        optimizer.zero_grad()
        gen_optimizer.zero_grad()

        # choose level 2/3 hierarchy, 20-class (gt) / 100-class classification (generated by labelgeneartor)
        train_loss1 = VGG16_model.model_fit(train_pred1, train_label[:, 2], pri=True, num_output=20)
        train_loss2 = VGG16_model.model_fit(train_pred2, train_pred3, pri=False, num_output=100)

        # compute cosine similarity between gradients from primary and auxiliary loss
        grads1 = torch.autograd.grad(torch.mean(train_loss1), VGG16_model.parameters(), retain_graph=True, allow_unused=True)
        grads2 = torch.autograd.grad(torch.mean(train_loss2), VGG16_model.parameters(), retain_graph=True, allow_unused=True)
        cos_mean = 0
        for l in range(len(grads1) - 8):  # only compute on shared representation (ignore task-specific fc-layers)
            grads1_ = grads1[l].view(grads1[l].shape[0], -1)
            grads2_ = grads2[l].view(grads2[l].shape[0], -1)
            cos_mean += torch.mean(F.cosine_similarity(grads1_, grads2_, dim=-1)) / (len(grads1) - 8)
        # cosine similarity evaluation ends here

        train_loss = torch.mean(train_loss1) + torch.mean(train_loss2)
        train_loss.backward()

        optimizer.step()

        train_predict_label1 = train_pred1.data.max(1)[1]
        train_acc1 = train_predict_label1.eq(train_label[:, 2]).sum().item() / batch_size

        cost[0] = torch.mean(train_loss1).item()
        cost[1] = train_acc1
        cost[2] = cos_mean
        k = k + 1
        avg_cost[index][0:3] += cost[0:3] / train_batch

    # evaluating training data (meta-training step, update on theta_2)
    cifar100_train_dataset = iter(cifar100_train_loader)
    for i in range(train_batch):
        train_data, train_label = cifar100_train_dataset.next()
        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = train_data.to(device), train_label.to(device)

        # reset optimizer with zero gradient
        optimizer.zero_grad()
        gen_optimizer.zero_grad()

        maxl.unrolled_backward(train_data, train_label[:, 2], scheduler.get_last_lr()[0], optimizer)
        gen_optimizer.step()

    # evaluate on test data
    VGG16_model.eval()
    with torch.no_grad():
        cifar100_test_dataset = iter(cifar100_test_loader)
        for i in range(test_batch):
            test_data, test_label = cifar100_test_dataset.next()
            test_label = test_label.type(torch.LongTensor)
            test_data, test_label = test_data.to(device), test_label.to(device)

            test_pred1, test_pred2 = VGG16_model(test_data)
            test_loss1 = VGG16_model.model_fit(test_pred1, test_label[:, 2], pri=True, num_output=20)

            test_predict_label1 = test_pred1.data.max(1)[1]
            test_acc1 = test_predict_label1.eq(test_label[:, 2]).sum().item() / batch_size

            cost[0] = torch.mean(test_loss1).item()
            cost[1] = test_acc1

            avg_cost[index][3:] += cost[0:2] / test_batch

    scheduler.step()
    gen_scheduler.step()
    print('EPOCH: {:04d} Iter {:04d} | TRAIN [LOSS|ACC.]: PRI {:.4f} {:.4f} COSSIM {:.4f} || TEST: {:.4f} {:.4f}'
          .format(index, k, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3],
                  avg_cost[index][4]))
