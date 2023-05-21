# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 16:18:41 2021

@author: Amartya Choudhury
"""

import torch.nn as nn
import torch
from torch_geometric.nn import ChebConv
from graph_utils import getGCNParams
from torch.nn.utils import weight_norm
from tcn import TemporalConvNet
import math
import numpy as np
from utils import buildGraphFromJson, getPMLevelsImputed, extractComponents, countParameters, normalize, setupCUDA, saveModel
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import mean_squared_error as mse
from visuals import plotPredictions

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

class GCLSTM(torch.nn.Module):
    def __init__(self, Th, Tp):
        super().__init__()
        self.gcn = ChebConv(Th, Th, 3)
        self.lstm = LSTM(2, 32, 2, 1, Tp)
    
    def forward(self, X, A):
        B = X.shape[0]
        N = X.shape[1]
        Th = X.shape[2]
        Ab = A.repeat(B, 1, 1) 
        
        b, idx, wt = getGCNParams(Ab)
        H = self.gcn(X, idx, wt, b)          #B x N x Th
        Xgcn = torch.cat((X, H), dim = -1)  #B x N x Th x 2
        Xgcn = Xgcn.view(B*N, Th, 2)        #BN x Th x 2
        Z = self.lstm(Xgcn)                 #BN x Tp x 1
        Z = Z.view(B, N, Z.shape[1])
        return Z

class  PolnData(torch.utils.data.Dataset):
    def __init__(self, Xh, Y, indices):
        self.indices = indices
        self.Xh = Xh[indices]
        self.Y = Y[indices]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.Xh[index], self.Y[index]

def train(model, Xh, Y, A, N, device, batch=32, epochs=400):
    Xh = torch.Tensor(Xh).to(device)
    Y = torch.Tensor(Y).to(device)
    A = torch.Tensor(A).to(device)

    indices = np.arange(Xh.shape[0])
    idxTrain, idxVal = train_test_split(indices, test_size = 0.2, random_state = 42)
    
    trainData = PolnData(Xh, Y, idxTrain)
    valData = PolnData(Xh, Y, idxVal)
    
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size = batch, shuffle = True)
    valLoader = torch.utils.data.DataLoader(valData, batch_size = batch, shuffle = False)
    
    mse = torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    
    fVloss= 0
    for e in range(epochs):
        loss = 0.0
        valLoss = 0.0
        for Xh, Y in trainLoader:
            Yp = model(Xh, A)
            lossT = mse(Yp, Y)
            optimizer.zero_grad()
            lossT.backward()
            optimizer.step()
            loss += lossT.item()
  
        with torch.set_grad_enabled(False):
                for Xh, Y in valLoader:
                    Yval = model(Xh, A)
                    lossV = mse(Yval, Y)
                    valLoss += lossV.item()
        
        epochLoss = loss / len(trainLoader)
        epochValLoss = loss/ len(valLoader)
        fVloss = epochValLoss
        print(f"Epoch: {e}, train_loss: {epochLoss}, validation_loss = {epochValLoss}")
    return model, fVloss

############################# MAIN ###############################

if __name__ == "__main__":
    #### TRAIN ####
    '''
    N, A, coordsDf = buildGraphFromJson('coords.json')
    trainPath = "PM10/train"
    X_train, F, T = getPMLevelsImputed(trainPath, coordsDf, N)
    
    X_train, scaler = normalize(X_train)
    X_train = X_train.reshape(N, 1, T)
    
    Tp = 12
    Th = 24
    Td = 12
    Tw = 24
    q = 24 * 4
    
    #extract weekly, hourly and recent components
    Xh, Xd, Xw, Y = extractComponents(X_train, Tp, Th, Td, Tw, q)
    
    Xh = Xh.reshape(Xh.shape[0], N, Th)
    Y =   Y.reshape(Y.shape[0], N, Tp)
    
    device = setupCUDA()
    gclstm = GCLSTM(Th, Tp)
    countParameters(gclstm)
    gclstm.to(device)
    
    batch = 32
    epochs = 400
    gclstm, fVloss = train(gclstm, Xh, Y, A, N, device, batch, epochs)
    
    fn = f"GCLSTM_Th{Th}_Tp{Tp}_hidden_32_layers_2_b{batch}_e{epochs}_vl{fVloss}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"
    saveModel(gclstm, fn, "models", "PM10", "GCLSTM")
    '''

    #### TEST ####
    Tp = 12
    Th = 24
    Td = 12
    Tw = 24
    q = 24 * 4
    device = setupCUDA()
    gclstm = GCLSTM(Th, Tp)
    countParameters(gclstm)
    gclstm.to(device)
    
    path = "models/PM10/GCLSTM/" + "GCLSTM_Th24_Tp12_hidden_32_layers_2_b32_e400_vl0.010422150230234755_18_13_18_24-07-21.pt"
    gclstm.load_state_dict(torch.load(path))
    testPath = "PM10/test"
    N, A, coordsDf = buildGraphFromJson('coords.json')
    X_test, F, T = getPMLevelsImputed(testPath, coordsDf, N)
    X_test, scaler = normalize(X_test)
    X_test = X_test.reshape(N, 1, T)
    Xh, Xd, Xw, Yt = extractComponents(X_test, Tp, Th, Td, Tw, q)
    Xh = Xh.reshape(Xh.shape[0], N, Th)
    Yt =   Yt.reshape(Yt.shape[0], N, Tp)
    Xh = torch.Tensor(Xh).to(device)
    A = torch.Tensor(A).to(device)
    
    
    total = Xh.shape[0]
    batch  = 64
    start = 0
    end = batch
    Yp = np.array([])
    
    while (start <= total ):
        Xh_b = Xh[start:end]
        Yp_b = gclstm(Xh_b, A).cpu().detach().numpy()
        if (Yp.size == 0):
            Yp = Yp_b
        else:
            Yp = np.vstack((Yp, Yp_b))
        start += batch
        end += batch
    
    Yp = Yp.reshape(-1, 1)
    Yp = scaler.inverse_transform(Yp)
    
    Yt = Yt.reshape(-1, 1)
    Yt = scaler.inverse_transform(Yt)
    
    Yp = Yp.reshape(total, N, Tp)
    Yt = Yt.reshape(total, N, Tp)
    
    Yp = np.transpose(Yp, (1, 0, 2))   # N x obs X Tp
    Yp = np.reshape(Yp, (Yp.shape[0], 1, Yp.shape[1] * Yp.shape[2]))
    
    Yt = np.transpose(Yt, (1, 0, 2))  
    Yt = np.reshape(Yt, (Yt.shape[0], 1, Yt.shape[1] * Yt.shape[2]))
    
    plotPredictions(Yt[:, :, 0:500], Yp[:, :, 0:500], coordsDf, N)
    print(mse(Yt[:, 0, :], Yp[:, 0, :]))  #1080