"""Dataset setting and data loader for CIFAR10."""


import torch
from torchvision import datasets, transforms

import params

import numpy as np


def get_cifar10(train, corruption=None):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    # dataset and data loader
    cifar10_dataset = datasets.CIFAR10(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    if corruption:
        corruption, level = corruption.split(',')
        if train:
            length = 50000
        else:
            length = 10000
        trainset_raw = np.load('../data/myCIFAR-10-C/CIFAR-10-C-trainval/train/%s.npy' %(corruption))[(int(level)-1)*length: int(level)*length]
        cifar10_dataset.data = trainset_raw

    cifar10_data_loader = torch.utils.data.DataLoader(
        dataset=cifar10_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return cifar10_data_loader
