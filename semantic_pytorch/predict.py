import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import time
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Function
from keras.utils.np_utils import *
import math

from collections import defaultdict
import os
import copy
from scipy import spatial
from keras.utils.np_utils import *
from sklearn import metrics, neighbors
import cv2
import csv
import ast
import json
import pandas as pd
import random
from PIL import Image
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable 

from collections import OrderedDict
import pandas as pd
from sklearn import metrics
from core.models.dfanet import DFANet
from config import Config
from core.models.deeplabv3_plus import DeepLabV3Plus
from core.models.pspnet import PSPNet
from operator import truediv

class_dict = dict()
class_dict['idx_unique'] = {}
pos =[]
if os.path.exists('out/colormap.txt'):
    with open('out/colormap.txt', 'r') as f:
        colormap = f.readline()
    pos = np.loadtxt('out/colormap.txt',dtype=float,delimiter=' ')
    class_dict['colormap'] = colormap
else:
    class_dict['colormap'] = []

device = torch.device('cuda')
 
#set trainFunction
def calc_loss(pred, target, metrics, criterion):
    loss = criterion(pred, target)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss
 
    
def print_metrics(metrics, epoch_samples, phase, epoch):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
 
    
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
 
def prepare_data(y_, num_class):
    def png2idx(data):
        # 将3通道的mask转换为单通道数组
        data = data.astype('uint32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return idx
 
    label_idx = png2idx(y_)
    if not os.path.exists('out/colormap.txt'):
        for idx in np.unique(label_idx):
            if idx not in class_dict['idx_unique'].keys():
                class_dict['idx_unique'][idx] = len(class_dict['idx_unique'])
                flag = 0
                for row in range(label_idx.shape[0]):
                    if flag == 1:
                        break
                    for col in range(label_idx.shape[1]):
                        if label_idx[row][col] == idx:
                            class_dict['colormap'].append(list(y_[row, col, :]))
                            flag = 1
                            break
 
    # 将三通道的colormap映射为idx
    cm2lbl = np.zeros(256**3)    # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
    if [224, 224, 192] in class_dict['colormap']:
        class_dict['colormap'].remove([224, 224, 192])
 
    if len(class_dict['colormap']) == num_class and not os.path.exists('out/colormap.txt'):
        with open('out/colormap.txt', 'w') as f:
            f.write(str(class_dict['colormap']))
    for i, cm in enumerate(class_dict['colormap']):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i    # 建立索引
    # 得到单维的mask
    mask_2D = cm2lbl[label_idx]
    mask_2D = mask_2D.astype(int)

    return mask_2D
 

def trans_label(label):
    class_list = pos
    color_label = np.zeros((label.shape[0], label.shape[1], 3))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            num = int(label[i][j])
            if num != 0 :
                num -=1
                color_label[i][j][0] = class_list[num][0]
                color_label[i][j][1] = class_list[num][1]
                color_label[i][j][2] = class_list[num][2]
    return color_label

def valid_process(model, device, val_pic, val_label):
    model.to(device)
    model.eval()
    print('success load model')
    
    all_account = 0
    all_MIou = 0

    with torch.no_grad():
        for i in range(len(val_pic)):
            pic = val_pic[i]
            label = val_label[i]
            color_pic = pic.cuda().data.cpu().numpy()
            pic = Variable(torch.unsqueeze(pic, dim=0).float(), requires_grad=False)
            label = Variable(torch.unsqueeze(label, dim=0).float(), requires_grad=False)
            inputs = pic.to(device)
            outputs = model(inputs)
            outputs = np.asarray(outputs)
            label = np.asarray(label)
            outputs = outputs[0]  

            a=outputs.cpu().numpy() 
            a = a[0,:,:,:]
            outputs = np.transpose(a, (1, 2, 0))
            label = np.transpose(label,(1,2,0))
            outputs = outputs.astype(int)

            label=np.squeeze(label)
            cal_outputs = np.zeros((outputs.shape[0], outputs.shape[1]))
            cal_outputs2=np.argmax(outputs,axis=2)
            account = outputs.shape[0]* outputs.shape[1]

            ac_pixel = 0
            wr_pixel = 0
            TP = 0
            TF = 0
            TP = 0
            TF = 0

            cal_outputs = cal_outputs2.flatten()
            label = label.flatten()

            cm = metrics.confusion_matrix(cal_outputs, label)
            cm = cm.astype(np.float32)
            FP = cm.sum(axis=0) - np.diag(cm)
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)

            list_diag = np.diag(cm)
            list_raw_sum = np.sum(cm,axis=1)

            each_acc = np.nan_to_num(truediv(list_diag,list_raw_sum))
            print("acc: ", each_acc)
            try:              
                MIou =  ( TP / (TP + FP + FN) + TN / (TN + FN + FP) ) / 2
                    
                MIou = np.mean(MIou)
                all_MIou += MIou
            except:
                pass

            all_account += np.mean(each_acc)
            color_outputs = trans_label(cal_outputs2)
            output_dir = 'out/result/predict/'
            if os.path.exists(output_dir) == False:
                os.makedirs(output_dir)            
            output_path = output_dir + str(i) + '.png'
            color_outputs = np.uint8(color_outputs)
            image = Image.fromarray(color_outputs) 
            image.save(output_path)              
               
    average_ac = all_account / len(val_pic)
    average_MIOU = all_MIou / len(val_pic)
    print('accuracy:', average_ac)
    print('MIOU:', average_MIOU)

def read_dataset(val_dataset_path):
    data = pd.read_csv(val_dataset_path).values
    
    val_pic = []
    val_mask = []

    for idx in range(len(data)):
        img_name = data[idx][0]
        lab_name = data[idx][1]

        label_path = lab_name.strip('\n')
        train_path = '' + img_name
        label_path = '' + label_path

        train = cv2.imread(train_path)
        pic_dir = 'out/result/pic3/'
        pic_path = pic_dir + str(idx) + '.png'
        cv2.imwrite(pic_path, train) #保存测试图片
        
        label_dir = 'out/result/label/'
        outlabel_path = label_dir + str(idx) + '.png'
        
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        train = np.asarray(train)

        label.flags.writeable = True
        label = np.asarray(label)
        
        label.flags.writeable = True
        label_seg = label
        
        color_label = trans_label(label)
        color_label = np.uint8(color_label)
        image = Image.fromarray(color_label)

        image.save(outlabel_path)
        train = np.transpose(train, (2, 0, 1))
        
        train = train.astype(np.float32)
        imgA = torch.from_numpy(train)
        imgB = torch.from_numpy(label)

        val_pic.append(imgA)
        val_mask.append(imgB)
        
    return val_pic, val_mask    
 
    
def Valid_DFA():   
    cfg=Config()
    model = DeepLabV3Plus(backbone='xception')
    state_dict = torch.load('model/fix_deeplab_v3_cc.pt')
    model.load_state_dict(state_dict)
    device = torch.device("cuda")

    val_dataset_path = '/openbayes/home/test.csv'
    val_pic, val_label = read_dataset(val_dataset_path)  
    valid_process(model, device, val_pic, val_label)
    
if __name__=='__main__':
    Valid_DFA()
    
    
