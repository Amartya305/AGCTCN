# -*- coding: utf-8 -*-
"""
Created on Wed May  5 03:18:55 2021

@author: Amartya Choudhury
"""

import torch.nn as nn
import torch
from torch_geometric.nn import ChebConv
from graph_utils import getGCNParams
from torch.nn.utils import weight_norm
from sklearn.model_selection import train_test_split
from datetime import datetime
import random
import numpy as np
from utils import *

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

    
class TemporalAtt(nn.Module):
    def __init__(self, N, C, T):
        super().__init__()
        self.Ve = nn.Parameter(torch.Tensor(T, T))
        self.be = nn.Parameter(torch.Tensor(T, T))
        self.U1 = nn.Parameter(torch.Tensor(N))
        self.U2 = nn.Parameter(torch.Tensor(C, N))
        self.U3 = nn.Parameter(torch.Tensor(C))
        nn.init.xavier_uniform_(self.Ve)
        nn.init.xavier_uniform_(self.be)
        nn.init.uniform_(self.U1)
        nn.init.xavier_uniform_(self.U2)    
        nn.init.uniform_(self.U3)
        
    def forward(self, X):                                  # B X N X C X T
        B = X.shape[0]
        xT = X.permute(0, 3, 2, 1)                         # B X T X C X N
        _E = torch.matmul(xT, self.U1)                     # B X T X C
        _E = torch.matmul(_E, self.U2)                     # B X T X N
        _E = torch.matmul(_E, torch.matmul(self.U3, X))    # B X T X T 
        _E = torch.add(_E, self.be.repeat(B, 1, 1))        # B X T X T
        sig = nn.Softmax(1)
        _E = sig(_E)
        _E = torch.matmul(self.Ve.repeat(B, 1, 1), _E)      
        E  = sig(_E)                                       # B X T X T
        return E

class STBlock(nn.Module):
    def __init__(self, N, T1, T2, K):
        super().__init__()
        self.T2 = T2
        self.sAtt = SpatialAtt(N, 1, T1)
        #self.tAtt = TemporalAtt(N, 1, T1)
        self.gcn = ChebConv(T1, T2, K)                  
        self.timeConv = nn.Conv1d(1, 1, K, padding = int((K - 1) / 2))
    
    def forward(self, X, A):                                                                   # B X N X 1 X T
        B = X.shape[0]
        N = X.shape[1]
        T = X.shape[3]
        
        S = self.sAtt(X)                                                                       # B X N X N    
        Abatch = A.repeat(B, 1, 1)                                                 
        A_hat = torch.mul(Abatch, S)                                                           # B X N X N
        
        #E = self.tAtt(X)                                                                       # B X T X T
        #E = E.repeat(1, N, 1)                                                                  # B X NT X T
        #E = E.view(B, N, T, T)                                                                 # B X N X T X T
        #X_hat = torch.matmul(X, E)                                                             # B X N X 1 X T  
        
        X_hat = X.view(B, N, T)                                                            # B x N x T
        batch, edgeIndex, edgeWeight = getGCNParams(A_hat)
        X_new = self.gcn(X_hat, edgeIndex, edgeWeight, batch)                                  # B x N x T2
        X_new = X_new.relu()                                                                   
        
        X_new = X_new.view(B*N, 1, self.T2)                                                    # BN X 1 x T2
        X_new = self.timeConv(X_new)                                                           # BN X 1 X T2
        X_new = X_new.relu()
        
        X_new = X_new.view(B, N, 1, self.T2)                                                   # B x N x 1 X T2        
        return X_new
        
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
    
class ASTGCN(nn.Module):
    def __init__(self, N, Th, Td, Tw, Tp):
        super().__init__()
        self.sptH1 = STBlock(N, Th, 64, 3)
        self.sptH2 = STBlock(N, 64, 32, 3)
        self.LinearH = nn.Linear(32, Tp)
     
        self.sptD1 = STBlock(N, Td, 64, 3)
        self.sptD2 = STBlock(N, 64, 32, 3)
        self.LinearD = nn.Linear(32, Tp)        
      
        self.sptW1 = STBlock(N, Tw, 64, 3)
        self.sptW2 = STBlock(N, 64, 32, 3)
        self.LinearW = nn.Linear(32, Tp)
        
        self.Fusion = Fusion(Tp)
    
    def forward(self, Xh, Xd, Xw, A):
        Yh = self.sptH1(Xh, A)
        Yh = self.sptH2(Yh, A)
        Yh = self.LinearD(Yh)
        Yh = Yh.relu()
        
        Yd = self.sptD1(Xd, A)
        Yd = self.sptD2(Yd, A)
        Yd = self.LinearD(Yd)
        Yd = Yd.relu()
        
        Yw = self.sptW1(Xw, A)
        Yw = self.sptW2(Yw, A)     
        Yw = self.LinearD(Yw)
        Yw = Yw.relu()
        
        Y = self.Fusion(Yh, Yd, Yw)
        return Y


class  PolnData(torch.utils.data.Dataset):
    def __init__(self, Xh, Xd, Xw, indices):
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

def setupCUDA():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    return device
    
def trainASTGCN(Xh, Xd, Xw, Y, A, N, Th, Td, Tw, Tp, device, batch = 32, epochs = 400):
    Xh = torch.Tensor(Xh).to(device)
    Xd = torch.Tensor(Xd).to(device)
    Xw = torch.Tensor(Xw).to(device)
    Y = torch.Tensor(Y).to(device)
    A = torch.Tensor(A).to(device)

    indices = np.arange(Xh.shape[0])
    idxTrain, idxVal = train_test_split(indices, test_size = 0.2, random_state = 42)
    
    trainData = PolnData(Xh, Xd, Xw, idxTrain)
    valData = PolnData(Xh, Xd, Xw, idxVal)
    
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size = batch, shuffle = True)
    valLoader = torch.utils.data.DataLoader(valData, batch_size = batch, shuffle = False)
    
    model = ASTGCN(N, Th, Td, Tw, Tp)
    countParameters(model)
    model.to(device)
    
    mse = torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    
    for e in range(epochs):
        loss = 0.0
        valLoss = 0.0
        for Xh, Xd, Xw, Y in trainLoader:
            Yp = model(Xh, Xd, Xw)
            lossT = mse(Yp, Y)
            optimizer.zero_grad()
            lossT.backward()
            optimizer.step()
            loss += lossT.item()
        
        with torch.set_grad_enabled(False):
                for Xh, Xd, Xw, Y in valLoader:
                    Yval = model(Xh, Xd, Xw)
                    lossV = mse(Yval, Y)
                    valLoss += lossV.item()
        
        epochLoss = loss / len(trainLoader)
        epochValLoss = loss/ len(valLoader)
        print(f"Epoch: {e}, train_loss: {epochLoss}, validation_loss = {epochValLoss}")
    
    #torch.save(model.state_dict(), name + datetime.now().strftime("%H_%M_%S_%d-%m-%y") + ".pt")
    
def test(model, X, Th, Td, Tp, q, nTrain, device, path = None):
    if path is not None:
        model.load_state_dict(torch.load(path))
    Xh_t, Xd_t, Xw_t, Y_t = extractComponents(X, Tp, Th, Td, Tw, q, t0 = nTrain - 1)
    
    Xh_t = getSingleFeature(Xh_t, 0)
    Xd_t = getSingleFeature(Xd_t, 0)
    Xw_t = getSingleFeature(Xw_t, 0)
    Y_t = getSingleFeature(Y_t, 0)
    
    Xh_t = torch.Tensor(Xh_t).to(device)
    Xd_t = torch.Tensor(Xd_t).to(device)
    Xw_t = torch.Tensor(Xw_t).to(device)
    
    Y_p = model(Xh_t, Xd_t, Xw_t).cpu().detach().numpy()

    Y_p = np.transpose(Y_pred, (1, 0, 2, 3))   # N x obs X 1 X Tp
    Y_p = np.reshape(Y_pred, (Y_pred.shape[0], 1, Y_pred.shape[1] * Y_pred.shape[3]))

    Y_t = np.transpose(Y_t, (1, 0, 2, 3))  
    Y_t = np.reshape(Y_t, (Y_t.shape[0], 1, Y_t.shape[1] * Y_t.shape[3]))
    
    return Y_p, Y_t

class Fusion2(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.W1 = nn.Parameter(torch.Tensor(T))
        self.W2 = nn.Parameter(torch.Tensor(T))
        nn.init.uniform_(self.W1)
        nn.init.uniform_(self.W2)
        
    def forward(self, Y1, Y2):                         # B x N x 1 x TP
        sum = torch.mul(self.W1, Y2)
        sum = torch.add(sum, torch.mul(self.W1, Y2))
        return sum                                         # B x N x 1 x TP
    
    
class ASTGCN_noWeekly(nn.Module):
    def __init__(self, N, Th, Td, Tp):
        super().__init__()
        self.sptH1 = STBlock(N, Th, 64, 3)
        self.sptH2 = STBlock(N, 64, 32, 3)
        self.LinearH = nn.Linear(32, Tp)
     
        self.sptD1 = STBlock(N, Td, 64, 3)
        self.sptD2 = STBlock(N, 64, 32, 3)
        self.LinearD = nn.Linear(32, Tp)        
              
        self.Fusion = Fusion2(Tp)
    
    def forward(self, Xh, Xd, A):
        Yh = self.sptH1(Xh, A)
        Yh = self.sptH2(Yh, A)
        Yh = self.LinearD(Yh)
        Yh = Yh.relu()
        
        Yd = self.sptD1(Xd, A)
        Yd = self.sptD2(Yd, A)
        Yd = self.LinearD(Yd)
        Yd = Yd.relu()
        
        
        Y = self.Fusion(Yh, Yd)
        return Y

class ASTGCN_noDaily(nn.Module):
    def __init__(self, N, Th, Tw, Tp):
        super().__init__()
        self.sptH1 = STBlock(N, Th, 64, 3)
        self.sptH2 = STBlock(N, 64, 32, 3)
        self.LinearH = nn.Linear(32, Tp)
     
        self.sptW1 = STBlock(N, Tw, 64, 3)
        self.sptW2 = STBlock(N, 64, 32, 3)
        self.LinearD = nn.Linear(32, Tp)        
              
        self.Fusion = Fusion2(Tp)
    
    def forward(self, Xh, Xw, A):
        Yh = self.sptH1(Xh, A)
        Yh = self.sptH2(Yh, A)
        Yh = self.LinearD(Yh)
        Yh = Yh.relu()
        
        Yw = self.sptW1(Xw, A)
        Yw = self.sptW2(Yw, A)
        Yw = self.LinearD(Yw)
        Yw = Yw.relu()
        
        
        Y = self.Fusion(Yh, Yw)
        return Y
    

class ASTGCN_noRecent(nn.Module):
    def __init__(self, N, Td, Tw, Tp):
        super().__init__()
        self.sptD1 = STBlock(N, Td, 64, 3)
        self.sptD2 = STBlock(N, 64, 32, 3)
        self.LinearH = nn.Linear(32, Tp)
     
        self.sptW1 = STBlock(N, Tw, 64, 3)
        self.sptW2 = STBlock(N, 64, 32, 3)
        self.LinearD = nn.Linear(32, Tp)        
              
        self.Fusion = Fusion2(Tp)
    
    def forward(self, Xd, Xw, A):
        Yd = self.sptD1(Xd, A)
        Yd = self.sptD2(Yd, A)
        Yd = self.LinearD(Yd)
        Yd = Yd.relu()
        
        Yw = self.sptW1(Xw, A)
        Yw = self.sptW2(Yw, A)
        Yw = self.LinearD(Yw)
        Yw = Yw.relu()
        
        
        Y = self.Fusion(Yd, Yw)
        return Y


class STBlock_noSatt(nn.Module):
    def __init__(self, N, T1, T2, K):
        super().__init__()
        self.T2 = T2
        #self.sAtt = SpatialAtt(N, 1, T1)
        #self.tAtt = TemporalAtt(N, 1, T1)
        self.gcn = ChebConv(T1, T2, K)                  
        self.timeConv = nn.Conv1d(1, 1, K, padding = int((K - 1) / 2))
    
    def forward(self, X, A):                                                                   # B X N X 1 X T
        B = X.shape[0]
        N = X.shape[1]
        T = X.shape[3]
        
        #S = self.sAtt(X)                                                                       # B X N X N    
        Abatch = A.repeat(B, 1, 1)                                                 
        #A_hat = torch.mul(Abatch, S)                                                           # B X N X N
        
        X_hat = X.view(B, N, T)                                                            # B x N x T
        batch, edgeIndex, edgeWeight = getGCNParams(Abatch)
        X_new = self.gcn(X_hat, edgeIndex, edgeWeight, batch)                                  # B x N x T2
        X_new = X_new.relu()                                                                   
        
        X_new = X_new.view(B*N, 1, self.T2)                                                    # BN X 1 x T2
        X_new = self.timeConv(X_new)                                                           # BN X 1 X T2
        X_new = X_new.relu()
        
        X_new = X_new.view(B, N, 1, self.T2)                                                   # B x N x 1 X T2        
        return X_new

class ASTGCN_noSatt(nn.Module):
    def __init__(self, N, Th, Td, Tw, Tp):
        super().__init__()
        self.sptH1 = STBlock_noSatt(N, Th, 64, 3)
        self.sptH2 = STBlock_noSatt(N, 64, 32, 3)
        self.LinearH = nn.Linear(32, Tp)
     
        self.sptD1 = STBlock_noSatt(N, Td, 64, 3)
        self.sptD2 = STBlock_noSatt(N, 64, 32, 3)
        self.LinearD = nn.Linear(32, Tp)        
      
        self.sptW1 = STBlock_noSatt(N, Tw, 64, 3)
        self.sptW2 = STBlock_noSatt(N, 64, 32, 3)
        self.LinearW = nn.Linear(32, Tp)
        
        self.Fusion = Fusion(Tp)
    
    def forward(self, Xh, Xd, Xw, A):
        Yh = self.sptH1(Xh, A)
        Yh = self.sptH2(Yh, A)
        Yh = self.LinearD(Yh)
        Yh = Yh.relu()
        
        Yd = self.sptD1(Xd, A)
        Yd = self.sptD2(Yd, A)
        Yd = self.LinearD(Yd)
        Yd = Yd.relu()
        
        Yw = self.sptW1(Xw, A)
        Yw = self.sptW2(Yw, A)     
        Yw = self.LinearD(Yw)
        Yw = Yw.relu()
        
        Y = self.Fusion(Yh, Yd, Yw)
        return Y

class STBlock_noTatt(nn.Module):
    def __init__(self, N, T1, T2, K):
        super().__init__()
        self.T2 = T2
        self.sAtt = SpatialAtt(N, 1, T1)
        #self.tAtt = TemporalAtt(N, 1, T1)
        self.gcn = ChebConv(T1, T2, K)                  
        self.timeConv = nn.Conv1d(1, 1, K, padding = int((K - 1) / 2))
    
    def forward(self, X, A):                                                                   # B X N X 1 X T
        B = X.shape[0]
        N = X.shape[1]
        T = X.shape[3]
        
        S = self.sAtt(X)                                                                       # B X N X N    
        Abatch = A.repeat(B, 1, 1)                                                 
        A_hat = torch.mul(Abatch, S)                                                           # B X N X N
        
        #E = self.tAtt(X)                                                                       # B X T X T
        #E = E.repeat(1, N, 1)                                                                  # B X NT X T
        #E = E.view(B, N, T, T)                                                                 # B X N X T X T
        #X_hat = torch.matmul(X, E)                                                             # B X N X 1 X T  
        
        X = X.view(B, N, T)                                                            # B x N x T
        batch, edgeIndex, edgeWeight = getGCNParams(A_hat)
        X_new = self.gcn(X, edgeIndex, edgeWeight, batch)                                  # B x N x T2
        X_new = X_new.relu()                                                                   
        
        X_new = X_new.view(B*N, 1, self.T2)                                                    # BN X 1 x T2
        X_new = self.timeConv(X_new)                                                           # BN X 1 X T2
        X_new = X_new.relu()
        
        X_new = X_new.view(B, N, 1, self.T2)                                                   # B x N x 1 X T2        
        return X_new

class ASTGCN_noTatt(nn.Module):
    def __init__(self, N, Th, Td, Tw, Tp):
        super().__init__()
        self.sptH1 = STBlock_noTatt(N, Th, 64, 3)
        self.sptH2 = STBlock_noTatt(N, 64, 32, 3)
        self.LinearH = nn.Linear(32, Tp)
     
        self.sptD1 = STBlock_noTatt(N, Td, 64, 3)
        self.sptD2 = STBlock_noTatt(N, 64, 32, 3)
        self.LinearD = nn.Linear(32, Tp)        
      
        self.sptW1 = STBlock_noTatt(N, Tw, 64, 3)
        self.sptW2 = STBlock_noTatt(N, 64, 32, 3)
        self.LinearW = nn.Linear(32, Tp)
        
        self.Fusion = Fusion(Tp)
    
    def forward(self, Xh, Xd, Xw, A):
        Yh = self.sptH1(Xh, A)
        Yh = self.sptH2(Yh, A)
        Yh = self.LinearD(Yh)
        Yh = Yh.relu()
        
        Yd = self.sptD1(Xd, A)
        Yd = self.sptD2(Yd, A)
        Yd = self.LinearD(Yd)
        Yd = Yd.relu()
        
        Yw = self.sptW1(Xw, A)
        Yw = self.sptW2(Yw, A)     
        Yw = self.LinearD(Yw)
        Yw = Yw.relu()
        
        Y = self.Fusion(Yh, Yd, Yw)
        return Y
    
class STBlock_noGCN(nn.Module):
    def __init__(self, N, T1, T2, K):
        super().__init__()
        self.T2 = T2
        self.sAtt = SpatialAtt(N, 1, T1)
        #self.tAtt = TemporalAtt(N, 1, T1)
        #self.gcn = ChebConv(T1, T2, K)
        self.dummy = nn.Linear(T1, T2)                  
        self.timeConv = nn.Conv1d(1, 1, K, padding = int((K - 1) / 2))
    
    def forward(self, X, A):                                                                   # B X N X 1 X T
        B = X.shape[0]
        N = X.shape[1]
        T = X.shape[3]
        
        S = self.sAtt(X)                                                                       # B X N X N    
        Abatch = A.repeat(B, 1, 1)                                                 
        A_hat = torch.mul(Abatch, S)                                                           # B X N X N
        
        #E = self.tAtt(X)                                                                       # B X T X T
        #E = E.repeat(1, N, 1)                                                                  # B X NT X T
        #E = E.view(B, N, T, T)                                                                 # B X N X T X T
        #X_hat = torch.matmul(X, E)                                                             # B X N X 1 X T  
        
        X_hat = X.view(B, N, T)                                                            # B x N x T
        #batch, edgeIndex, edgeWeight = getGCNParams(A_hat)
        #X_new = self.gcn(X, edgeIndex, edgeWeight, batch)                                  # B x N x T2
        X_new = self.dummy(X_hat)
        X_new = X_new.relu()                                                                   
        
        X_new = X_new.view(B*N, 1, self.T2)                                                    # BN X 1 x T2
        X_new = self.timeConv(X_new)                                                           # BN X 1 X T2
        X_new = X_new.relu()
        
        X_new = X_new.view(B, N, 1, self.T2)                                                   # B x N x 1 X T2        
        return X_new

class ASTGCN_noGCN(nn.Module):
    def __init__(self, N, Th, Td, Tw, Tp):
        super().__init__()
        self.sptH1 = STBlock_noGCN(N, Th, 64, 3)
        self.sptH2 = STBlock_noGCN(N, 64, 32, 3)
        self.LinearH = nn.Linear(32, Tp)
     
        self.sptD1 = STBlock_noGCN(N, Td, 64, 3)
        self.sptD2 = STBlock_noGCN(N, 64, 32, 3)
        self.LinearD = nn.Linear(32, Tp)        
      
        self.sptW1 = STBlock_noGCN(N, Tw, 64, 3)
        self.sptW2 = STBlock_noGCN(N, 64, 32, 3)
        self.LinearW = nn.Linear(32, Tp)
        
        self.Fusion = Fusion(Tp)
    
    def forward(self, Xh, Xd, Xw, A):
        Yh = self.sptH1(Xh, A)
        Yh = self.sptH2(Yh, A)
        Yh = self.LinearD(Yh)
        Yh = Yh.relu()
        
        Yd = self.sptD1(Xd, A)
        Yd = self.sptD2(Yd, A)
        Yd = self.LinearD(Yd)
        Yd = Yd.relu()
        
        Yw = self.sptW1(Xw, A)
        Yw = self.sptW2(Yw, A)     
        Yw = self.LinearD(Yw)
        Yw = Yw.relu()
        
        Y = self.Fusion(Yh, Yd, Yw)
        return Y
    
class STBlock_noTime(nn.Module):
    def __init__(self, N, T1, T2, K):
        super().__init__()
        self.T2 = T2
        self.sAtt = SpatialAtt(N, 1, T1)
        #self.tAtt = TemporalAtt(N, 1, T1)
        self.gcn = ChebConv(T1, T2, K)                  
        #self.timeConv = nn.Conv1d(1, 1, K, padding = int((K - 1) / 2))
    
    def forward(self, X, A):                                                                   # B X N X 1 X T
        B = X.shape[0]
        N = X.shape[1]
        T = X.shape[3]
        
        S = self.sAtt(X)                                                                       # B X N X N    
        Abatch = A.repeat(B, 1, 1)                                                 
        A_hat = torch.mul(Abatch, S)                                                           # B X N X N
        
        #E = self.tAtt(X)                                                                       # B X T X T
        #E = E.repeat(1, N, 1)                                                                  # B X NT X T
        #E = E.view(B, N, T, T)                                                                 # B X N X T X T
        #X_hat = torch.matmul(X, E)                                                             # B X N X 1 X T  
        
        X_hat = X.view(B, N, T)                                                            # B x N x T
        batch, edgeIndex, edgeWeight = getGCNParams(A_hat)
        X_new = self.gcn(X_hat, edgeIndex, edgeWeight, batch)                                  # B x N x T2
        X_new = X_new.relu()                                                                   
        
        #X_new = X_new.view(B*N, 1, self.T2)                                                    # BN X 1 x T2
        #X_new = self.timeConv(X_new)                                                           # BN X 1 X T2
        #X_new = X_new.relu()
        
        X_new = X_new.view(B, N, 1, self.T2)                                                   # B x N x 1 X T2        
        return X_new

class ASTGCN_noTime(nn.Module):
    def __init__(self, N, Th, Td, Tw, Tp):
        super().__init__()
        self.sptH1 = STBlock_noTime(N, Th, 64, 3)
        self.sptH2 = STBlock_noTime(N, 64, 32, 3)
        self.LinearH = nn.Linear(32, Tp)
     
        self.sptD1 = STBlock_noTime(N, Td, 64, 3)
        self.sptD2 = STBlock_noTime(N, 64, 32, 3)
        self.LinearD = nn.Linear(32, Tp)        
      
        self.sptW1 = STBlock_noTime(N, Tw, 64, 3)
        self.sptW2 = STBlock_noTime(N, 64, 32, 3)
        self.LinearW = nn.Linear(32, Tp)
        
        self.Fusion = Fusion(Tp)
    
    def forward(self, Xh, Xd, Xw, A):
        Yh = self.sptH1(Xh, A)
        Yh = self.sptH2(Yh, A)
        Yh = self.LinearD(Yh)
        Yh = Yh.relu()
        
        Yd = self.sptD1(Xd, A)
        Yd = self.sptD2(Yd, A)
        Yd = self.LinearD(Yd)
        Yd = Yd.relu()
        
        Yw = self.sptW1(Xw, A)
        Yw = self.sptW2(Yw, A)     
        Yw = self.LinearD(Yw)
        Yw = Yw.relu()
        
        Y = self.Fusion(Yh, Yd, Yw)
        return Y                