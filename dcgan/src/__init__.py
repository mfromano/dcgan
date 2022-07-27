'''
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import pandas as pd
import pandera as pa
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
import random

seed = 9999

random.seed(seed)
torch.manual_seed(seed)

IMG_LOCATION = "/home/mfromano/Research/celeba/CelebA/Img/"
SEED = 999
BATCH_SIZE = 128
WORKERS = 2
CHANNELS = 3
N_FEAT_LATENT = 100
IMG_DIM = 64
N_FEAT_MAP_G = 64
N_FEAT_MAP_D = 64
EPOCHS = 5
LR = 0.0002
Adam_beta = 0.5
N_GPU = 1

DATASET = dset.ImageFolder(
        root=IMG_LOCATION,
        transform=transforms.Compose([
                transforms.Resize(IMG_DIM),
                transforms.CenterCrop(IMG_DIM),
                transforms.ToTensor(),
                transforms.Normalize(
                        (0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5)
                ),
        ])
)

DATALOADER = torch.utils.data.DataLoader(
        DATASET,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = WORKERS
)
