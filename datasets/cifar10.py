"""Dataset setting and data loader for CIFAR10."""


import torch
from torchvision import datasets, transforms

import params


def get_cifar10(train):
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

    cifar10_data_loader = torch.utils.data.DataLoader(
        dataset=cifar10_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return cifar10_data_loader
