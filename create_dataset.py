from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

import os
import glob
import pickle
import numpy as np
import scipy.misc
#import matplotlib.pyplot as plt
import warnings
import torch

class CIFAR100(Dataset):
    def __init__(self, data_path, train=True, transform=None, auxiliary=None, file_index=1):
        """
        Args:
            data_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.auxiliary = auxiliary
        self.train     = train

        # Read the data file
        if train:
            self.data_path = data_path + '/cifar-100-python/train'
        else:
            self.data_path = data_path + '/cifar-100-python/test'
        with open(self.data_path, 'rb') as fo:
            self.data_info = pickle.load(fo, encoding='latin1')
        fo.close()

        # Calculate len
        self.data_len = len(self.data_info['data'])

        # First column contains the image paths
        self.image_arr = self.data_info['data'].reshape([self.data_len, 3, 32, 32])
        self.image_arr = self.image_arr.transpose((0, 2, 3, 1))  # convert to HWC

        # 10 Class, build dict from 20 class:
        class_10 = {0: 1, 1: 2, 2: 5, 3: 6, 4: 5, 5: 6, 6: 6, 7: 3, 8: 0, 9: 7,
                    10: 8, 11: 0, 12: 1, 13: 3, 14: 4, 15: 0, 16: 2, 17: 5, 18: 9, 19: 9}
        # 3 Class, build dict from 10 class:
        class_3 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}

        # Second column is the labels
        self.label_fine   = self.data_info['fine_labels']
        self.label_coarse = self.data_info['coarse_labels']
        self.label_c10    = np.vectorize(class_10.get)(self.label_coarse)
        self.label_c3     = np.vectorize(class_3.get)(self.label_c10)

        if train:
            np.save(data_path + '/cifar100_gt.npy', [self.label_c3, self.label_c10, self.label_coarse, self.label_fine])

            if auxiliary is not None and not os.path.isfile(data_path + '/cifar100_aux_{:d}.npy'.format(file_index)):
                self.data_aux = np.zeros(len(self.label_fine))
                label_all = np.load(data_path + '/cifar100_gt.npy')
                # fine the correct auxiliary class
                for aux_index in range(4):
                    if label_all[aux_index, :].max() == (len(auxiliary) - 1):
                        self.aux_index = aux_index
                        break
                    if aux_index is not 3:
                        continue
                    else:
                        warnings.warn("Please define the correct hierarchy to be in: 3, 10, 20, 100.")

                # randomly assign a class from predefined hierarchy
                for label in range(len(self.auxiliary)):
                    index = [i for i, x in enumerate(label_all[self.aux_index, :]) if x == label]
                    random_class = np.random.randint(auxiliary[label], size=len(index)) + np.sum(auxiliary[:label])
                    self.data_aux[index] = random_class
                np.save(data_path + '/cifar100_aux_{:d}.npy'.format(file_index), self.data_aux)

            if os.path.isfile(data_path + '/cifar100_aux_{:d}.npy'.format(file_index)):
                self.data_aux = np.load(data_path + '/cifar100_aux_{:d}.npy'.format(file_index))
                self.aux_index = 2

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]

        # Open image
        img_as_img = Image.fromarray(single_image_name)

        # Transform image to tensor
        if self.transform is not None:
            img_as_img = self.transform(img_as_img)
        else:
            img_as_img = self.to_tensor(img_as_img)

        # Get label(class) of the image (from coarse to fine)
        label = np.zeros(4)
        label[-1] = self.label_fine[index]
        label[-2] = self.label_coarse[index]
        label[-3] = self.label_c10[index]
        label[-4] = self.label_c3[index]

        if self.auxiliary is None or not self.train:
            return img_as_img, label, index
        else:
            return img_as_img, np.array([label[self.aux_index], self.data_aux[index]]), index

    def __len__(self):
        return self.data_len


class CIFAR10(Dataset):
    def __init__(self, data_path, type, transform=None, auxiliary=None, file_index=0):
        """
        Args:
            data_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.auxiliary = auxiliary
        self.type = type

        train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ]

        val_list = [
            ['test_batch', '40351d587109b95175f43aff81a1287e'],
        ]

        # Read the data file
        if type == 'train':
            self.data = []
            self.labels = []
            for fentry in train_list:
                file = data_path + '/cifar-10-batches-py/' + fentry[0]
                fo = open(file, 'rb')
                entry = pickle.load(fo, encoding='latin1')
                self.data.append(entry['data'])
                self.labels += entry['labels']
                fo.close()

            self.data = np.concatenate(self.data)
        if type == 'val':
            file = data_path + '/cifar-10-batches-py/' + val_list[0][0]
            fo = open(file, 'rb')
            entry = pickle.load(fo, encoding='latin1')

            self.data = entry['data']
            self.labels = entry['labels']
            fo.close()
        if type == 'test':
            self.data = np.load(data_path + '/cifar10.1_v6_data.npy') 
            self.labels = np.load(data_path + '/cifar10.1_v6_labels.npy')

        self.data_len = len(self.data)
        if type != 'test':
            self.data = self.data.reshape((self.data_len, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if auxiliary is not None and not os.path.isfile(data_path + '/cifar10_aux{:d}.npy'.format(file_index)):
            self.data_aux = np.zeros(len(self.labels))
            label_all = self.labels
            # fine the correct auxiliary class
            if len(auxiliary) != 10:
                warnings.warn("Please define the correct primary hierarchy to be: 10.")

            # randomly assign a class from predefined hierarchy
            for label in range(len(self.auxiliary)):
                index = [i for i, x in enumerate(label_all) if x == label]
                random_class = np.random.randint(auxiliary[label], size=len(index)) + np.sum(auxiliary[:label])
                self.data_aux[index] = random_class
            np.save(data_path + '/cifar10_aux{:d}.npy'.format(file_index), self.data_aux)

        if os.path.isfile(data_path + '/cifar10_aux{:d}.npy'.format(file_index)):
            self.data_aux = np.load(data_path + '/cifar10_aux{:d}.npy'.format(file_index))

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.data[index]

        # Open image
        img_as_img = Image.fromarray(single_image_name)

        # Transform image to tensor
        if self.transform is not None:
            img_as_img = self.transform(img_as_img)
        else:
            img_as_img = self.to_tensor(img_as_img)

        # Get label(class) of the image (from coarse to fine)
        label = self.labels[index]

        if self.type != 'train' or (self.auxiliary is None):
            return img_as_img, label, index
        else:
            return img_as_img, np.array([label, self.data_aux[index]]), index

    def __len__(self):
        return self.data_len



class NYUv2(Dataset):
    def __init__(self, data_path, type, task, transform=None):
        """
        Args:
            data_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.type = type
        self.task = task

        # Read the data file
        if type == 'train':
            self.file = data_path + '/train'
            #self.data_len = 2975
            self.data_len = 795

        if type == 'val':
            self.file = data_path + '/val'
            #self.data_len = 500
            self.data_len = 654



    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = np.load(self.file + '/image/{:d}.npy'.format(index))

        # Transform image to tensor
        if self.transform is not None:
            img_as_img = self.transform(single_image_name)
        else:
            img_as_img = self.to_tensor(single_image_name)

        # Get label(class) of the image (from coarse to fine)
        if self.task == 'semantic':
            label = np.load(self.file + '/label/{:d}.npy'.format(index))
        if self.task == 'depth':
            label = np.load(self.file + '/depth/{:d}.npy'.format(index))
        if self.task == 'normal':
            label = np.load(self.file + '/normal2/{:d}.npy'.format(index))

        return img_as_img.type(torch.float), label

    def __len__(self):
        return self.data_len


