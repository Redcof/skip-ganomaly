"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from lib.data.datasets import get_cifar_anomaly_dataset
from lib.data.datasets import get_mnist_anomaly_dataset
from lib.data.sixray_sd3_anomaly_dataset import SIXraySD3AnomalyDataset


class Data:
    """ Dataloader containing train and valid sets.
    """
    
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid


##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """
    
    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)
    
    ## CIFAR
    if opt.dataset in ['cifar10']:
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        train_ds = CIFAR10(root='./data', train=True, download=True, transform=transform)
        valid_ds = CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_ds, valid_ds = get_cifar_anomaly_dataset(train_ds, valid_ds, train_ds.class_to_idx[opt.abnormal_class])
    
    ## MNIST
    elif opt.dataset in ['mnist']:
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        
        train_ds = MNIST(root='./data', train=True, download=True, transform=transform)
        valid_ds = MNIST(root='./data', train=False, download=True, transform=transform)
        train_ds, valid_ds = get_mnist_anomaly_dataset(train_ds, valid_ds, int(opt.abnormal_class))
    elif opt.dataset in ['sixray_sd3']:
        train_ds = SIXraySD3AnomalyDataset(data_dir=opt.dataroot, nc=opt.nc, imsize=opt.isize, split="train")
        valid_ds = SIXraySD3AnomalyDataset(data_dir=opt.dataroot, nc=opt.nc, imsize=opt.isize, split="test")
    # FOLDER
    else:
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        
        train_ds = ImageFolder(os.path.join(opt.dataroot, 'train'), transform)
        valid_ds = ImageFolder(os.path.join(opt.dataroot, 'test'), transform)
    
    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)
    
    return Data(train_dl, valid_dl)
