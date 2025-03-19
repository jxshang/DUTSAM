import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
import time
import torch
import random
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def get_hz_array(df,index,len,cols,n_hz):
    hz_array = []
    for col in cols:
        hz_array.append(df[col][index,:len*n_hz])#*n_hz 若统一输入长度需删掉

    hz_array = torch.tensor(hz_array.copy(),dtype=torch.float32)
    return hz_array

def get_array(df, param_list, index):
    array = []
    for col in param_list:
        array.append(df[col][index])
    array = torch.tensor(array.copy(),dtype=torch.float32)
    return array

class QARDataset(Dataset):

    def __init__(self, config, data, Y):
        super(QARDataset, self).__init__()
        self.Y = Y 
        self.data_dict = data
        self.param_list = config.param_list

    def get_samples(self, index):

        #one_hz_array = get_hz_array(self.data_dict, index, time_length, one_cols, 1)
        #two_hz_array = get_hz_array(self.data_dict, index, time_length, two_cols, 2)
        #four_hz_array = get_hz_array(self.data_dict, index, time_length, four_cols, 4)
        #eight_hz_array = get_hz_array(self.data_dict, index, time_length, eight_cols, 8)
        #
        #X = torch.cat([one_hz_array, two_hz_array, four_hz_array, eight_hz_array], 0)
        X = get_array(self.data_dict, self.param_list, index)
        return X

    def __getitem__(self, index):

        label = self.Y[index]
        #直接为类别
        target = torch.tensor(label, dtype=torch.long)
        #one-hot标签
        #target = np.zeros(num_classes)
        #target[label] = 1
        #target = torch.Tensor(target)
        
        #维度是(N, H, W)
        X = self.get_samples(index)
        return X, target

    def __len__(self):
        return len(self.Y) 


def get_data_loader(config,all_train_dict, all_test_dict, Y_train, Y_test):
    train_dataset = QARDataset(config,all_train_dict, Y_train)
    test_dataset = QARDataset(config,all_test_dict, Y_test)
    print(len(train_dataset), len(test_dataset))

    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # all_dataloader = DataLoader(all_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    #test_loader = DataLoader(test_dataset, batch_size=170, shuffle=False)
    return train_loader,test_loader
# train_loader,test_loader = get_data_loader(config,all_train_dict,all_test_dict,Y_train,Y_test)