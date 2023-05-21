# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:36:56 2021

@author: amart
"""

import torch.nn as nn
import torch
from torch_geometric.nn import ChebConv
from graph_utils import getGCNParams
from torch.nn.utils import weight_norm
from tcn import TemporalConvNet
import math
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, output_seq):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers
        self.output_seq = output_seq
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda').requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda').requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -self.output_seq : , :].contiguous())
        return out

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
        self.tAtt = TemporalAtt(N, 1, T1)
        self.gcn = ChebConv(T1, T2, K)
        #self.lstm = LSTM(1, 32, 2, 1,  12)                  

    
    def forward(self, X, A):                                                                   # B X N X 1 X T
        B = X.shape[0]
        N = X.shape[1]
        T = X.shape[3]
        
        S = self.sAtt(X)                                                                       # B X N X N    
        Abatch = A.repeat(B, 1, 1)                                                 
        A_hat = torch.mul(Abatch, S)                                                           # B X N X N
        
        E = self.tAtt(X)                                                                       # B X T X T
        E = E.repeat(1, N, 1)                                                                  # B X NT X T
        E = E.view(B, N, T, T)                                                                 # B X N X T X T
        X_hat = torch.matmul(X, E)                                                             # B X N X 1 X T  
        
        X_hat = X_hat.view(B, N, T)                                                            # B x N x T
        batch, edgeIndex, edgeWeight = getGCNParams(Abatch)
        X_new = self.gcn(X_hat, edgeIndex, edgeWeight, batch)                                  # B x N x T2
        X_new = X_new.relu()                                                                   

        #X_new = X_new.view(B*N, self.T2, 1)                                                    # BN X T2 X 1
        #X_new = self.lstm(X_new)                                                               # BN X T2 X 1
        #X_new = X_new.relu()
        
        X_new = X_new.view(B, N, 1, self.T2)                                                   # B x N x 1 X T2        
        #X_new = X_new.view(B, N, 1, 12)
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
    
class ASTGCNLSTM(nn.Module):
    def __init__(self, N, Th, Td, Tw, Tp):
        super().__init__()
        self.sptH1 = STBlock(N, Th, 64, 3)
        self.sptH2 = STBlock(N, 64, 32, 3)
     
        self.sptD1 = STBlock(N, Td, 64, 3)
        self.sptD2 = STBlock(N, 64, 32, 3)     
      
        self.sptW1 = STBlock(N, Tw, 64, 3)
        self.sptW2 = STBlock(N, 64, 32, 3)
        self.Tp = Tp
        self.N  = N
        self.lstm = LSTM(3, 32, 2, 1, Tp)
        #self.Fusion = Fusion(Tp)
    
    def forward(self, Xh, Xd, Xw, A):
        B  = Xh.shape[0]
        Yh = self.sptH1(Xh, A)
        Yh = self.sptH2(Yh, A)
#        Yh = self.LinearD(Yh)
#        Yh = Yh.relu()
        
        Yd = self.sptD1(Xd, A)
        Yd = self.sptD2(Yd, A)
#        Yd = self.LinearD(Yd)
#        Yd = Yd.relu()
        
        Yw = self.sptW1(Xw, A)
        Yw = self.sptW2(Yw, A)     
#        Yw = self.LinearD(Yw)
#        Yw = Yw.relu()
        Yh = Yh.view(B*self.N, 32, 1)
        Yd = Yd.view(B*self.N, 32, 1)
        Yw = Yw.view(B*self.N, 32, 1)
        
        Yf = torch.cat((Yh, Yd, Yw), dim = -1)
        Yf = self.lstm(Yf)
        Yf = Yf.view(B, self.N, 1, self.Tp)
        #Y = self.Fusion(Yh, Yw, Yd)
        return Yf