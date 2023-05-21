# -*- coding: utf-8 -*-
"""
Created on Tue May 24 22:50:11 2022

@author: amart
"""

import torch.nn as nn
import torch
from torch_geometric.nn import ChebConv
from graph_utils import getGCNParams
from torch.nn.utils import weight_norm
from tcn import TemporalConvNet
import torch.optim as optim
import math
import numpy as np
import optuna
from optuna.trial import TrialState

from utils import buildGraphFromJson, getPMLevelsImputed, extractComponents, countParameters, normalize
from astgcn import ASTGCN
from sklearn.model_selection import train_test_split
from datetime import datetime
import cProfile
import pstats
import torch
import random
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import sys


def setupCUDA():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.benchmark=True
    np.random.seed(1)
    return device

class  PolnData(torch.utils.data.Dataset):
    def __init__(self, Xh, Xd, Xw, Y, indices):
        self.indices = indices
        self.Xh = Xh[indices]
        self.Xd = Xd[indices]
        self.Xw = Xw[indices]
        self.Y = Y[indices]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.Xh[index], self.Xd[index], self.Xw[index], self.Y[index]

class SpatialAtt(nn.Module):
    def __init__(self, N, C, T):
        super().__init__()
        self.Vs = nn.Parameter(torch.Tensor(N, N))
        self.bs = nn.Parameter(torch.Tensor(N, N))
        self.W1 = nn.Parameter(torch.Tensor(T))
        self.W2 = nn.Parameter(torch.Tensor(C, T))
        self.W3 = nn.Parameter(torch.Tensor(C))
        nn.init.xavier_uniform_(self.Vs)
        nn.init.xavier_uniform_(self.bs)
        nn.init.uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)    
        nn.init.uniform_(self.W3)
        
    def forward(self, X):                              # B x N x C x T
        B = X.shape[0]
        W1x = torch.matmul(X, self.W1)                 # B X N X C
        W3x = torch.matmul(self.W3, X)                 # B X N X T  
        W3xT = W3x.permute(0, 2, 1)                    # B X T X N
        W1xW2 = torch.matmul(W1x, self.W2)             # B X N X T
        _S = torch.matmul(W1xW2, W3xT)                 # B X N X N
        _S = torch.add(_S, self.bs.repeat(B, 1, 1))    # B X N X N
        sig =  nn.Softmax(1)                            
        _S = sig(_S)
        _S = torch.matmul(self.Vs.repeat(B, 1, 1), _S)
        S =  sig(_S)                                   # B X N X N
        return S

class Fusion(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.Wh = nn.Parameter(torch.Tensor(T))
        self.Wd = nn.Parameter(torch.Tensor(T))
        self.Ww = nn.Parameter(torch.Tensor(T))
        nn.init.uniform_(self.Wh)
        nn.init.uniform_(self.Wd)
        nn.init.uniform_(self.Ww)
        
    def forward(self, Yh, Yd, Yw):                         # B x N x 1 x TP
        sum = torch.mul(self.Wh, Yh)
        sum = torch.add(sum, torch.mul(self.Wd, Yd))
        sum = torch.add(sum, torch.mul(self.Ww, Yw))
        return sum                                         # B x N x 1 x TP
    
    
class STBlock(nn.Module):
    def __init__(self, N, A, T1, T2, K):
        super().__init__()
        self.T2 = T2
        self.sAtt = SpatialAtt(N, 1, T1)
        self.gcn = ChebConv(T1, T2, K)                  
        self.timeConv = nn.Conv1d(1, 1, 3, padding = 1)
        self.A = A
        self.K = K
        
    def forward(self, X):                                                                   # B X N X 1 X T
        B = X.shape[0]
        N = X.shape[1]
        T = X.shape[3]
        S = self.sAtt(X)
        A = self.A                                                                      # B X N X N    
        Abatch = A.repeat(B, 1, 1)                                                 
        A_hat = torch.mul(Abatch, S)                                                           # B X N X N
 
        X = X.view(B, N, T)                                                                    # B x N x T
        batch, edgeIndex, edgeWeight = getGCNParams(A_hat)
        X_new = self.gcn(X, edgeIndex, edgeWeight, batch)                                      # B x N x T2
        X_new = X_new.relu()                                                                   
        X_new = X_new.view(B*N, 1, self.T2)                                                    # BN X 1 x T2
        X_new = self.timeConv(X_new)
        X_new = X_new.relu()
        X_new = X_new.view(B, N, 1, self.T2)                                                   # B x N x 1 X T2        
        return X_new
'''
class ASTGCN(nn.Module):
    def __init__(self, N, A, Th, Td, Tw, Tp, trial):
        super().__init__()
        n_layers = trial.suggest_int("n_layers", 1, 3)
        sptH = []
        sptD = []
        sptW = []  
        
        in_features = Th
        for i in range(n_layers):
            out_features = trial.suggest_int("SptH{}".format(i), 4, 128)
            K = trial.suggest_int("SptH_k{}".format(i), 2, 4)
            p = trial.suggest_float("SptH_dropout{}".format(i), 0.1, 0.5)
            sptH.append(STBlock(N, A, in_features, out_features, K))
            sptH.append(nn.Dropout(p))
            in_features = out_features
        sptH.append(nn.Linear(out_features, Tp))
        
        in_features = Td
        for i in range(n_layers):
            out_features = trial.suggest_int("SptD{}".format(i), 4, 128)
            K = trial.suggest_int("SptD_k{}".format(i), 1, 4)
            p = trial.suggest_float("SptD_dropout{}".format(i), 0.1, 0.5)
            sptD.append(STBlock(N, A, in_features, out_features, K))
            sptD.append(nn.Dropout(p))
            in_features = out_features
        sptD.append(nn.Linear(out_features, Tp))
        
        in_features = Tw
        for i in range(n_layers):
            out_features = trial.suggest_int("SptW{}".format(i), 4, 128)
            K = trial.suggest_int("SptW_k{}".format(i), 1, 4)
            p = trial.suggest_float("SptW_dropout{}".format(i), 0.1, 0.5)
            sptW.append(STBlock(N, A, in_features, out_features, K))
            sptW.append(nn.Dropout(p))
            in_features = out_features
        sptW.append(nn.Linear(out_features, Tp))

        self.sptH = nn.Sequential(*sptH)
        self.sptD = nn.Sequential(*sptD)
        self.sptW = nn.Sequential(*sptW)
        self.Fusion = Fusion(Tp)
    
    def forward(self, Xh, Xd, Xw):
        Yh = self.sptH(Xh)
        Yd = self.sptD(Xd)
        Yw = self.sptW(Xw)
        Yh = Yh.relu()
        Yd = Yd.relu()
        Yw = Yw.relu()
        Y = self.Fusion(Yh, Yw, Yd)
        return Y
'''
class ASTGCN(nn.Module):
    def __init__(self, N, A, Th, Td, Tw, Tp, trial):
        super().__init__()
        n_layers = trial.suggest_int("n_layers", 1, 3)
        sptH = []
        sptD = []
        sptW = []  
        K = trial.suggest_int("K", 2, 4)
        p = trial.suggest_categorical("Dropout", [0.1, 0.2, 0.3, 0.4, 0.5])
            

        hidden = []
        for i in range(n_layers):
            hidden.append(trial.suggest_categorical("Hidden{}".format(i), [4, 8, 16, 32, 64, 128, 256]))
        
        in_features = Th
        for i in range(n_layers):
            out_features = hidden[i]
            sptH.append(STBlock(N, A, in_features, out_features, K))
            sptH.append(nn.Dropout(p))
            in_features = out_features
        sptH.append(nn.Linear(out_features, Tp))
        
        in_features = Td
        for i in range(n_layers):
            out_features = hidden[i]
            sptD.append(STBlock(N, A, in_features, out_features, K))
            sptD.append(nn.Dropout(p))
            in_features = out_features
        sptD.append(nn.Linear(out_features, Tp))
        
        in_features = Tw
        for i in range(n_layers):
            out_features = hidden[i]
            sptW.append(STBlock(N, A, in_features, out_features, K))
            sptW.append(nn.Dropout(p))
            in_features = out_features
        sptW.append(nn.Linear(out_features, Tp))

        self.sptH = nn.Sequential(*sptH)
        self.sptD = nn.Sequential(*sptD)
        self.sptW = nn.Sequential(*sptW)
        self.Fusion = Fusion(Tp)
    
    def forward(self, Xh, Xd, Xw):
        Yh = self.sptH(Xh)
        Yd = self.sptD(Xd)
        Yw = self.sptW(Xw)
        Yh = Yh.relu()
        Yd = Yd.relu()
        Yw = Yw.relu()
        Y = self.Fusion(Yh, Yw, Yd)
        return Y
    
def objective(trial):
    N, A, coordsDf = buildGraphFromJson('coords.json')
    trainPath = "PM10/train"
    X_train, F, T = getPMLevelsImputed(trainPath, coordsDf, N)
    X_train = X_train.reshape(N, 1, T)
    X_train = X_train[:, :, 0:2000]
    Tp = 12
    Th = 24
    Td = 12
    Tw = 24
    q = 24 * 4
    
    #extract weekly, hourly and recent components
    Xh, Xd, Xw, Y = extractComponents(X_train, Tp, Th, Td, Tw, q)
    device = setupCUDA()
    A = torch.Tensor(A).to(device)
    astgcn = ASTGCN(N, A, Th, Td, Tw, Tp, trial).to(device)
    # Generate the optimizers.
    #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(astgcn.parameters(), lr=lr)
    batch = trial.suggest_categorical("batch", [4, 8, 16, 32, 64])
    
    Xh = torch.Tensor(Xh).to(device)
    Xd = torch.Tensor(Xd).to(device)
    Xw = torch.Tensor(Xw).to(device)
    Y = torch.Tensor(Y).to(device)


    indices = np.arange(Xh.shape[0])
    idxTrain, idxVal = train_test_split(indices, test_size = 0.1, random_state = 42)
    
    trainData = PolnData(Xh, Xd, Xw, Y, idxTrain)
    valData = PolnData(Xh, Xd, Xw, Y, idxVal)
    
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size = batch, shuffle = True)
    valLoader = torch.utils.data.DataLoader(valData, batch_size = batch, shuffle = False)
    
    mse = torch.nn.MSELoss()
    epochs = 100
    print(f"========Trial {trial._trial_id} params: {trial.params}===========")
    countParameters(astgcn)
    for e in range(epochs):
        loss = 0.0
        valLoss = 0.0
        for Xh, Xd, Xw, Y in trainLoader:
            Yp = astgcn(Xh, Xd, Xw)
            lossT = mse(Yp, Y)
            optimizer.zero_grad()
            lossT.backward()
            optimizer.step()
            loss += lossT.item()
            
        with torch.set_grad_enabled(False):
                for Xh, Xd, Xw, Y in valLoader:
                    Yval = astgcn(Xh, Xd, Xw)
                    lossV = mse(Yval, Y)
                    valLoss += lossV.item()
        epochTrainLoss = loss  / len(valLoader)
        epochValLoss = valLoss/ len(valLoader)
        trial.report(epochValLoss, e)
        print(f" Trial: {trial._trial_id} Epoch: {e}, train_loss: {epochTrainLoss}, validation_loss = {epochValLoss}")
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return epochValLoss

if __name__ == "__main__":
    sys.stdout = open("optuna.log", "w")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    #sys.stdout.close()