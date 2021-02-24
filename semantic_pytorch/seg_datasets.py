import os
import cv2
import csv
import ast
import json
import torch
import numpy as np
import pandas as pd
import random
from PIL import Image
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from keras.utils.np_utils import *
import pandas


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.len = SegDataset._read_image_ids(self.is_train)

    def __len__(self):
        return len(self.len)

    @staticmethod
    def _read_image_ids(is_train):

        if is_train:
            data = pd.read_csv('/openbayes/home/train.csv').values        
            random.seed(1)
            random.shuffle(data) 
            return data        
        
        else:
            data = pd.read_csv('/openbayes/home/train.csv').values        
            random.seed(1)
            random.shuffle(data)          
            return data      

        
    def __getitem__(self, idx):
        train_path, label_path= self.len[idx][0], self.len[idx][1]
        label_path = label_path.strip('\n')
        train_path = '' + train_path
        label_path = '' + label_path

        train = cv2.imread(train_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label.flags.writeable = True
        train = np.asarray(train)
        label = np.asarray(label)
        label.flags.writeable = True
        label_seg = label

        train = np.transpose(train, (2, 0, 1))
        train = train.astype(np.float32)
        imgA = torch.from_numpy(train)
        imgB = torch.FloatTensor(label_seg).long()
        
        item = {'A': imgA, 'B': imgB}
        return item