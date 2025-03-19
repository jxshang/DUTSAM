import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score,roc_auc_score
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.optimizer import Optimizer, required

def train_epoch(config, model, optimizer, criterion, train_loader):
    model.train()
    train_l_sum = 0.0
    acc_sum = 0
    n = 0

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    #for inputs, target in tqdm(train_loader):
    for inputs, target in train_loader:

        inputs = inputs.to(config.device, dtype=torch.float)
        target = target.to(config.device)
        
        optimizer.zero_grad()
        
        output, _, _= model(inputs)

        #交叉熵 target=[batch,1]
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()

        train_l_sum += loss.item() * target.shape[0]
        acc_sum += (output.argmax(dim=1) == target).sum().item()
        predict = torch.max(output, 1)[1].cpu().numpy()
        labels = target.cpu().numpy()

        n += target.shape[0]
        predict_all = np.append(predict_all, predict)
        labels_all = np.append(labels_all, labels)
    
    f1 = f1_score(labels_all, predict_all)
    recall = recall_score(labels_all, predict_all)
    precision = precision_score(labels_all, predict_all)
    
    return train_l_sum / n, acc_sum / n, f1, recall, precision



def val_epoch(config, model, criterion, data_loader):
    model.eval()
    val_l_sum = 0.0
    acc_sum = 0
    n = 0
    
    predict_all = np.array([], dtype=int) 
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for inputs, target in data_loader:
            inputs = inputs.to(config.device)
            target = target.to(config.device)
            
            output, _, _ = model(inputs)

            loss = criterion(output, target.long())

            val_l_sum += loss.item() * target.shape[0]
            acc_sum += (output.argmax(dim=1) == target).sum().item()
            predict = torch.max(output, 1)[1].cpu().numpy()
            labels = target.cpu().numpy()            
      
            n += target.shape[0]
            predict_all = np.append(predict_all, predict)
            labels_all = np.append(labels_all, labels)


    f1 = f1_score(labels_all, predict_all)
    recall = recall_score(labels_all, predict_all)
    precision = precision_score(labels_all, predict_all)

    return val_l_sum / n, acc_sum / n, f1, recall, precision



def training(config, model, train_loader, val_loader=None):

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    model = model.to(config.device)
    print("***************trainging**************")

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_acc, train_f1, train_recall, train_precision = train_epoch(config, model, optimizer, criterion, train_loader)
        if(val_loader != None):
            val_loss, val_acc, val_f1, val_recall, val_precision = val_epoch(config, model, criterion, val_loader)

            print('#epoch:%02d train_loss:%f train_acc:%f train_f1:%f train_recall:%f train_precision:%f\n' \
                            'val_loss:%f val_acc:%f val_f1:%f val_recall:%f val_precision:%f\n'
                    % (epoch, train_loss, train_acc, train_f1, train_recall, train_precision,
                        val_loss, val_acc, val_f1, val_recall, val_precision))
        else:
            print('#epoch:%02d train_loss:%f train_acc:%f train_f1:%f train_recall:%f train_precision:%f\n'
                    % (epoch, train_loss, train_acc, train_f1, train_recall, train_precision))


    #torch.save(model.state_dict(), "/home/qar/Project/Jupyter/ZhangRX/state_dict.pth")