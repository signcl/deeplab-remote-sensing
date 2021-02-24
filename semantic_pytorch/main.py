import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from train import Trainer
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.nn import L1Loss
import torch.optim as optim
from torch.optim import lr_scheduler
from config import Config
from loss import CrossEntropyLoss2d
# from loss import Softmax
from seg_datasets import SegDataset
from core.models.deeplabv3_plus import DeepLabV3Plus
from core.models.pspnet import PSPNet
from core.models.ccnet import CCNet

if __name__=='__main__':
    cfg=Config()
    train_dataset, val_dataset = SegDataset(is_train=True), SegDataset(is_train=False)
    train_loader = DataLoader(train_dataset,
                       batch_size = 2,
                       shuffle = True,
                       num_workers = 8,
                       drop_last = True)
    val_loader =  DataLoader(val_dataset,
                       batch_size = 4,
                       shuffle = True,
                       num_workers = 8,
                       drop_last = True)
    
    net = DeepLabV3Plus(backbone = 'xception')
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(
    net.parameters(), lr=0.05, momentum=0.9,weight_decay=0.00001)  #select the optimizer

    lr_fc=lambda iteration: (1-iteration/400000)**0.9

    exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer,lr_fc,-1)
    
    trainer = Trainer('training', optimizer, exp_lr_scheduler, net, cfg, './log')
    #trainer.load_weights(trainer.find_last())
    trainer.train(train_loader, val_loader, criterion, 600)
    trainer.evaluate(val_loader)
    print('Finished Training')