import torch as t
import torchvision.transforms as transforms
import torchvision

import numpy as np

from typing import Tuple


def determine_normalization(dataset: str, data_dir: str) -> Tuple[float]:
    """
    Calculates the per-channel mean and std pixel values. Used to z-score when calling transforms.Normalize().
    Args:
        dataset: Name of given dataset. Some datasets have more than 1 channel, so processing is different.
        data_dir: Directory of dataset.
    Returns:
        Per-channel mean and std of dataset. Also, global min and max pixel values.
    """
    if dataset == "cifar10":
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        # print(vars(train_set))
        print(train_set.train_data.shape)
        mean = train_set.train_data.mean(axis=(0, 1, 2)) / 255
        std = train_set.train_data.std(axis=(0, 1, 2)) / 255
        min = train_set.train_data.min() / 255
        max = train_set.train_data.max() / 255

    elif dataset == "cifar100":
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        # print(vars(train_set))
        print(train_set.train_data.shape)
        mean = np.mean(train_set.train_data, axis=(0, 1, 2)) / 255
        std = np.std(train_set.train_data, axis=(0, 1, 2)) / 255
        min = train_set.train_data.min() / 255
        max = train_set.train_data.max() / 255

    elif dataset == "mnist":
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        # print(vars(train_set))
        print(list(train_set.train_data.size()))
        mean = train_set.train_data.float().mean() / 255
        std = train_set.train_data.float().std() / 255
        min = train_set.train_data.min() / 255
        max = train_set.train_data.max() / 255

    return mean, std, min, max


def load_mnist(args) -> Tuple:
    """
    Loads the MNIST dataset from specified directory, otherwise downloads it into directory.
    Args:
        args: Arguments from an ArgumentParser.
    Returns:
        DataLoaders for train and test set, as well as stats describing the size of the dataset.
    """
    mean, std, min, max = determine_normalization("mnist", args.data_path)

    mean = mean.unsqueeze(0)
    std = std.unsqueeze(0)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = torchvision.datasets.MNIST(root=args.data_path, train=True, download=True, transform=train_transform)
    train_loader = t.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers, pin_memory=True)

    test_set = torchvision.datasets.MNIST(root=args.data_path, train=False, download=True, transform=test_transform)
    test_loader = t.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=True)

    return train_loader, test_loader


def load_cifar10(args) -> Tuple:
    """
    Loads the Cifar10 dataset from specified directory, otherwise downloads it into directory.
    Args:
        args: Arguments from an ArgumentParser.
    Returns:
        DataLoaders for train and test set, as well as stats describing the size of the dataset.
    """
    mean, std, min, max = determine_normalization("cifar10", args.data_path)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
    train_loader = t.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
    test_loader = t.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=True)

    return train_loader, test_loader


class Image(object):
    def __init__(self, args):
        if args.dataset == 'cifar10':
            train_loader, test_loader = load_cifar10(args)
        elif args.dataset == 'mnist':
            train_loader, test_loader = load_mnist(args)
        else:
            raise NotImplementedError(f'Unknown dataset: {args.dataset}')

        self.train = train_loader
        self.valid = test_loader
        self.test = self.valid
