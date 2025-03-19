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
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import random
from sklearn.metrics import f1_score,recall_score,precision_score
import warnings
from utils.tools import *
from utils.Datagenerator import *
from utils.TrainUtils import *
from models.DUTSAM import *


warnings.filterwarnings("ignore")

#还差能量
"""
param_list = ['WIN_ALG','WIN_CRS','IAS','GS','N11','TLA1','LDGNOS','ALT_STD','RADIO_LH','RADIO_RH',
              'VRTG','IVV','ROLL','ROLL_CMD','PITCH','PITCH_CMD','GW','RUDD']


param_list = ['WIN_ALG','WIN_CRS','GS','N11','TLA1','LDGNOS','ALT_STD','RADIO_LH',
              'IVV','ROLL','ROLL_CMD','PITCH','PITCH_CMD','GW','RUDD']

aggregatedParamList = ['WIN_ALG','WIN_CRS','GS','N11','TLA1','LDGNOS','ALT_STD','RADIO_LH',
              'IVV','ROLL','PITCH','GW','RUDD']
            
one_cols = ['WIN_ALG','WIN_CRS','IAS','GS','N11','TLA1','ALT_STD','IVV','GW']
two_cols = ['ROLL','RUDD']
four_cols = ['LDGNOS','RADIO_LH','RADIO_RH','PITCH']
eight_cols = ['ROLL_CMD','PITCH_CMD']
"""
"""
one_cols = ['WIN_ALG','WIN_CRS','IAS','GS','VAPP','N11','N12','TLA1','TLA2','DME1','DME2','FLAP_PL','FLAP_PR','ALT_QNH','ALT_STD','IVV','GW']
two_cols = ['ROLL','RUDD']
four_cols = ['LDGL','LDGR','LDGNOS','RADIO_LH','RADIO_RH','PITCH']
eight_cols = ['ROLL_CMD','PITCH_CMD']    
"""

class Config:
    
    #实验设置参数 
    source_path = "./dataset/A320/hupsampling/" 
    param_list = ['WIN_ALG','WIN_CRS','IAS','GS','N11','TLA1',
              'LDGNOS','ALT_QNH','ALT_STD','RADIO_LH',
              'IVV','ROLL','ROLL_CMD','PITCH','PITCH_CMD','GW','RUDD']
#     param_list = ['ALT_STD','RADIO_LH','IVV','IAS','LDGNOS','PITCH_CMD','ROLL_CMD','RUDD',
#       'WIN_CRS','WIN_ALG','N11','TLA1',]

    time_length = 30
    lr = 1e-3


#     param_len = [time_len*1 for i in range(17)]+[time_len*2 for i in range(2)]+[time_len*4 for i in range(6)]+[time_len*8 for i in range(2)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    #模型自身参数

    #变量数
    n_features = len(param_list)
    #drop_rate
    #drop_prob = 0.2
    temporalDim = 64
    temporalKernel = 1
    variableDim = 64
    variableKernel = 3

    num_classes = 2


    #模型训练参数
    batch_size=256
    num_epochs = 50

    average=False
    relate = False
config = Config()



seed = 183

setup_seed(seed)

seeds = [34, 87956, 168, 472, 979]

all_train_dict, all_test_dict, Y_train, Y_test = get_time_forward_datadict(config.param_list, time_forward=30-config.time_length, is_scale=True, scale_type="Quantile")

train_loader, test_loader = get_data_loader(config,all_train_dict, all_test_dict, Y_train, Y_test)

model = dualAttention(config)

training(config, model, train_loader, test_loader)