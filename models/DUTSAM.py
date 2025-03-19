import pandas as pd
import numpy as np
import math
import os
import sys
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import random
from sklearn.metrics import f1_score,recall_score,precision_score
import warnings

warnings.filterwarnings("ignore")

class temporalAttention(nn.Module):
    def __init__(self, config):
        super(temporalAttention, self).__init__()
        self.device = config.device
        self.time_length = config.time_length
        self.n_features = config.n_features

        self.cnnList = nn.ModuleList([nn.Conv1d(1, config.temporalDim, config.temporalKernel) for _ in range(self.time_length)])

        self.W1 = nn.ParameterList([nn.Parameter(torch.randn(config.temporalDim, 1).to(self.device)) for _ in range(self.time_length)])
        self.b1 = nn.ParameterList([nn.Parameter(torch.randn(1, self.n_features).to(self.device)) for _ in range(self.time_length)])

        self.W2 = nn.Parameter(torch.randn(config.temporalDim, 1).to(self.device))
        self.b2 = nn.Parameter(torch.randn(1, self.time_length).to(self.device))

        self.dropout = nn.Dropout(p=0.3)

        self.relu = nn.ReLU()
        #这里的激活函数待确定
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        

    #x的维度是(N, n_features, time_length)
    #取index:index+1是为了取一行的数据的同时保持维度不变
    def forward(self, x):
        result = []
        variAtten = []

        for index in range(self.time_length):
            singleTemporal = self.cnnList[index](x[:, :, index:index+1].transpose(1, 2)).transpose(1, 2)
            Vattention = torch.matmul(singleTemporal, self.W1[index]).transpose(1, 2) + self.b1[index]
            
            Vattention = self.dropout(Vattention)
            Vattention = self.sigmoid(Vattention)
            
            #Vattention = self.softmax(Vattention)
            
            #Vattention的维度是(N, 1, n_features)
            #singleTemporal的维度是(N, n_features, temporalDim)
            #得到的结果是(N, 1, temporalDim) -> (N, temporalDim)
            result.append(torch.bmm(Vattention, singleTemporal).squeeze(1))
            variAtten.append(Vattention.squeeze(1))
        
        #variAtten经过stack后的维度为(N, n_features, time_length)
        variAtten = torch.stack(variAtten, dim=2)
        result = torch.stack(result, dim=1)
        #result的维度是(N, time_length, temporalDim)

        #result = self.dropout(result)
        
        tempAtten = torch.matmul(result, self.W2).transpose(1, 2) + self.b2
        
        tempAtten = self.dropout(tempAtten)
        tempAtten = self.sigmoid(tempAtten)
        #tempAtten的维度是(N, 1, time_length)

        #tempAtten = self.softmax(tempAtten)
        
        result = torch.bmm(tempAtten, result)
        #attention的维度为(N, n_features, time_length)
        return result.squeeze(), (variAtten * tempAtten).squeeze()


class variableAttention(nn.Module):
    def __init__(self, config):
        super(variableAttention, self).__init__()
        
        self.cnnList = nn.ModuleList()
        self.W1 = nn.ParameterList()
        self.b1 = nn.ParameterList()
        self.device = config.device
        self.time_length = config.time_length
        self.n_features = config.n_features

        for _ in range(self.n_features):
            #根据计算，要保持去除右边那些padding产生的数字以后的长度等于原始长度，需要kernel size = padding + 1
            self.cnnList.append(nn.Conv1d(1, config.variableDim, config.variableKernel, padding=config.variableKernel-1))
            self.W1.append(nn.Parameter(torch.randn(config.variableDim, 1).to(self.device)))
            self.b1.append(nn.Parameter(torch.randn(1, self.time_length).to(self.device)))

        self.W2 = nn.Parameter(torch.randn(config.variableDim, 1).to(self.device))
        self.b2 = nn.Parameter(torch.randn(1, self.n_features).to(self.device))

        self.dropout = nn.Dropout(p=0.3)

        self.relu = nn.ReLU()
        #这里的激活函数待确定
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    #x的维度是(N, n_features, time_length)
    #取index:index+1是为了取一行的数据的同时保持维度不变
    def forward(self, x):
        result = []
        tempAtten = []

        for index in range(self.n_features):
            singleVariable = self.cnnList[index](x[:, index:index+1, :])[:, :, :self.time_length].transpose(1, 2)
            Tattention = torch.matmul(singleVariable, self.W1[index]).transpose(1, 2) + self.b1[index]
            
            Tattention = self.dropout(Tattention)
            Tattention = self.sigmoid(Tattention)
            
            #Tattention = self.softmax(Tattention)
            
            #Tattention的维度是(N, 1, time_length)
            #singleVariable的维度是(N, time_length, variableDim)
            #得到的结果是(N, 1, variableDim) -> (N, variableDim)
            result.append(torch.bmm(Tattention, singleVariable).squeeze(1))
            tempAtten.append(Tattention.squeeze(1))

        #tempAtten经过stack后的维度为(N, n_features, time_length)
        tempAtten = torch.stack(tempAtten, dim=1)
        result = torch.stack(result, dim=1)
        #result的维度是(N, n_features, variableDim)

        #result = self.dropout(result)
        
        variAtten = torch.matmul(result, self.W2).transpose(1, 2) + self.b2
        
        variAtten = self.dropout(variAtten)
        variAtten = self.sigmoid(variAtten)
        
        #variAtten的维度是(N, 1, n_features)

        #variAtten = self.softmax(variAtten)
        
        result = torch.bmm(variAtten, result)
        #attention的维度为(N, n_features, time_length)
        return result.squeeze(), (tempAtten * variAtten.transpose(1, 2)).squeeze()


class dualAttention(nn.Module):
    def __init__(self, config):
        super(dualAttention, self).__init__()
        self.num_classes = config.num_classes
        self.temporalAtten = temporalAttention(config)
        self.variableAtten = variableAttention(config)

        self.dropout = nn.Dropout(p=0.3)

        self.fc = nn.Linear(config.temporalDim + config.variableDim, self.num_classes)
        #self.fc = nn.Linear(variableDim, num_classes)

    def forward(self, x):
        a, atten1 = self.temporalAtten(x)
        b, atten2 = self.variableAtten(x)
        #a的维度是(N, temporalDim)，b的维度是(N, variableDim)

        #print(temp)
        #print(vari)
        a = self.dropout(a)
        b = self.dropout(b)

        return self.fc(torch.cat((a, b), dim=-1)), atten1, atten2
        #return self.fc(b), atten1, atten2
        
    def predict_proba(self, x):
        # 对于二分类问题，我们使用 Sigmoid 将输出转换为概率
        logits = self.forward(x)
        return torch.sigmoid(logits)  # 对每个类别的概率进行预测
    
