import numpy as np
import os
import time
import torch
import random
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,StandardScaler,QuantileTransformer
import warnings
import math
warnings.filterwarnings('ignore')
"""
数据处理等 工具函数
"""


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True
     random.seed(seed)

def samebyAverage(data, control_size):
    result = []
    for i in range(0, len(data), control_size):
        avl = np.mean(data[i : i+control_size])
        result.append(avl)
    return result
    
def get_T_cols(cols):
    need_cols = []
    for col in cols:
        if "T" in col:
            if col!="Time":
                need_cols.append(col)
    return need_cols



# 对原数据处理得到触地前time_forward的数据 并将其平均化为30-time_forward的长度 得到前30-time_forward的数据(包含)
def get_time_forward_datadict(param_list, time_forward=2, is_scale=True, scale_type="Z_score", rebuild=False):

    if(rebuild):
        #判断文件夹是否存在
        if not os.path.exists("/home/qar/Project/Jupyter/LiXQ2/DUTSAM/dataset/"+str(time_forward)):
            os.makedirs("/home/qar/Project/Jupyter/LiXQ2/DUTSAM/dataset/"+str(time_forward))


        #对每个文件处理得到相应的数据
        source = "/home/qar/Project/Jupyter/LiXQ2/DUTSAM/dataset/A320/hupsampling/"
        files = os.listdir(source)
        for file in tqdm(files):
            
            df = pd.read_csv(source+file)
            col = df.columns.tolist()
            frequency = int(len(col)/50) #得到频率数
            time, label = df["Time"].tolist(), df["Label"].tolist()
            tmin = (int(math.ceil(np.min(time)/frequency))-1)*frequency #统一不同尺度
            i = 0
            data = []
            for t in time:
                t = int(t)
                sample = df.iloc[i][t-tmin:t+1-time_forward*frequency].tolist() #包含接地那一刻
                sample_p = samebyAverage(sample,frequency) # 每nhz 求平均 
                i+=1
                data.append(sample_p)

            data_df = pd.DataFrame(data)
#             print("data shape:", data_df.shape)  # 确保数据维度正确
#             print("Expected columns:", 30 - time_forward)
            data_df.columns = ["T_{}".format(i) for i in range(30-time_forward)]
            data_df["Label"] = label
            data_df.to_csv("/home/qar/Project/Jupyter/LiXQ2/DUTSAM/dataset/"+str(time_forward)+"/"+file,index=False)
            """
            df = pd.read_csv(source+file)
            col = df.columns.tolist()
            frequency = int(len(col)/50) #得到频率数
            time, label = df["Time"].tolist(), df["Label"].tolist()
            i = 0
            data = []
            for t in time:
                t = int(t)
                if(t < 30 * frequency):
                    sample = df.iloc[i][:30*frequency-time_forward*frequency].tolist()
                else:
                    sample = df.iloc[i][t-30*frequency:t-time_forward*frequency].tolist() #包含接地那一刻
                i+=1
                data.append(sample)
            
            if(frequency != 1):
                avg = nn.AvgPool1d(kernel_size=frequency)
                data = avg(torch.Tensor(data).unsqueeze(0)).squeeze()
                data_df = pd.DataFrame(data.tolist())
                #此时的数据长度都是30
            else:
                data_df = pd.DataFrame(data)
            data_df.columns = ["T_{}".format(i) for i in range(30-time_forward)]
            data_df["Label"] = label
            data_df.to_csv("./dataset/"+str(time_forward)+"/"+file,index=False)"""
    

    all_train_dict = {}
    all_test_dict = {}

    #对train、test里同一属性的数据进行标准化
    path = "/home/qar/Project/Jupyter/LiXQ2/DUTSAM/dataset/"+str(time_forward)
    neg_sampled_index = None
    indices = None
    flag = True

    for param in tqdm(param_list):
        train = pd.read_csv(path+"/train_"+param+".csv")
        train['flag'] = 0
        test = pd.read_csv(path+"/test_"+param+".csv")
        test['flag'] = 1
        Y_train = train['Label'].values
        Y_test = test['Label'].values
        data = pd.concat((train,test))
        del train,test
        need_cols = get_T_cols(data.columns)
        if is_scale:
            #若需要标准化
            if param in param_list:
                if scale_type=="Z_score":
                    transform = StandardScaler()
                elif scale_type=="MaxMin":
                    transform = MinMaxScaler(feature_range=(0, 1))
                else:
                    transform = QuantileTransformer()
                data[need_cols] = transform.fit_transform(data[need_cols])
    
        train,test = data[data['flag']==0],data[data['flag']==1]
        del data
        all_train_dict[param] = train[need_cols].values
        #all_test_dict[param] = test[need_cols].values
        
        pos_index = np.where(Y_test == 1)
        neg_index = np.where(Y_test == 0)
        if(flag):
            pos, indices = np.unique(test[need_cols].values[pos_index], axis=0, return_index=True)
            print(indices)
            flag = False
        else:
            pos = test[need_cols].values[pos_index][indices]
            
        n = pos.shape[0]
        
        #这下面的是下采样负样本（正常着陆的）
        #if(neg_sampled_index == None):
        #    neg_sampled_index = random.sample(range(len(neg_index[0])), n)
        #neg = test[need_cols].values[neg_index][neg_sampled_index]
        
        #all_test_dict[param] = np.concatenate((pos, neg), axis=0)
        #Y_test = np.concatenate((np.ones(n, dtype=int), np.zeros(n, dtype=int)))
        #结束
        

        #这是不对负样本进行下采样的代码
        neg = test[need_cols].values[neg_index]

        all_test_dict[param] = np.concatenate((pos, neg), axis=0)
        Y_test = np.concatenate((np.ones(n, dtype=int), np.zeros(neg.shape[0], dtype=int)))
        #结束

    return all_train_dict, all_test_dict, Y_train, Y_test