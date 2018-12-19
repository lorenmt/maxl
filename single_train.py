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

from Config.config import config
from Model.vgg16 import VGG16


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
    VGG16 = VGG16(device)


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
            # train_data size: (batch_size, channels, weight, height)
            # train_label size: (batch_size, TODO 4? why?)

            train_label = train_label.type(torch.LongTensor)  # TODO why not done in loader itself?
            train_data, train_label = train_data.to(device), train_label.to(device)  # TODO why not done in loader itself?
            train_pred1 = VGG16(train_data)  # dynamic graph -> reinitializes the object; returns output of forward function -> softmax
            optimizer.zero_grad()  # empty gradients (make them zero)
            train_loss1 = VGG16.compute_focal_loss(train_pred1, train_label[:, 2], 20)  # TODO why 20?, why 2? -> BAD!
            train_loss = torch.mean(train_loss1)

            train_loss.backward()  # backpropagate error
            optimizer.step()  # gradient step

            train_predict_label1 = train_pred1.data.max(1)[1].to(device)  # get indices of predicted class (the index with the highest softmax score)

            train_acc1 = train_predict_label1.eq(train_label[:, 2]).sum().item() / batch_size  # .item() is equivalent to .asscalar() in numpy
            cost[0] = torch.mean(train_loss1).item()
            cost[1] = train_acc1

            k = k + 1
            avg_cost[epoch][0:4] += cost / n_train_batches  # TODO why exactly 0:4?

        # evaluating test data
        with torch.no_grad():
            cifar100_test_dataset = iter(cifar100_test_loader)
            for i in range(n_test_batches):
                test_data, test_label, _ = cifar100_test_dataset.next()  # get next test batch
                test_label = test_label.type(torch.LongTensor)  # do torch operation
                test_data, test_label = test_data.to(device), test_label.to(device)  # do torch operation
                test_pred1 = VGG16(test_data)
                test_loss1 = VGG16.compute_focal_loss(test_pred1, test_label[:, 2], 20)  # TOOD weird

                test_predict_label1 = test_pred1.data.max(1)[1]

                test_acc1 = test_predict_label1.eq(test_label[:, 2]).sum().item() / batch_size
                cost[0] = torch.mean(test_loss1).item()
                cost[1] = test_acc1

                avg_cost[epoch][4:] += cost / n_test_batches

        print('Epoch: {:04d} Iteration: {:04d} | CIFAR100-TRAIN: {:.4f} {:.4f}  || '
              'CIFAR100-TEST: {:.4f} {:.4f} '
              .format(epoch, k, avg_cost[epoch][0], avg_cost[epoch][1],
                      avg_cost[epoch][4], avg_cost[epoch][5]))  # TODO why 4?

    #torch.save(VGG16.state_dict(), 'model_weights/cifar10_human_old')
    #np.save('loss/cifar10_human_old.npy', avg_cost)
