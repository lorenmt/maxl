from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

import os
import glob
import pickle
import numpy as np
import scipy.misc
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
            return img_as_img, label
        else:
            return img_as_img, np.array([label[self.aux_index], self.data_aux[index]])

    def __len__(self):
        return self.data_len
