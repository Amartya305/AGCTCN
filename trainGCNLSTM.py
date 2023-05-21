# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 15:15:21 2021

@author: amart
"""

from utils import buildGraphFromJson, getPMLevelsImputed, extractComponents, countParameters, normalize, setupCUDA, saveModel
from gcnLSTM import ASTGCNLSTM
from sklearn.model_selection import train_test_split
from datetime import datetime
import cProfile
import pstats
import torch
import random
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from visuals import plotPredictions

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

def train(model, Xh, Xd, Xw, Y, A, N, device, batch=32, epochs=400):
    Xh = torch.Tensor(Xh).to(device)
    Xd = torch.Tensor(Xd).to(device)
    Xw = torch.Tensor(Xw).to(device)
    Y = torch.Tensor(Y).to(device)
    A = torch.Tensor(A).to(device)

    indices = np.arange(Xh.shape[0])
    idxTrain, idxVal = train_test_split(indices, test_size = 0.2, random_state = 42)
    
    trainData = PolnData(Xh, Xd, Xw, Y, idxTrain)
    valData = PolnData(Xh, Xd, Xw, Y, idxVal)
    
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size = batch, shuffle = True)
    valLoader = torch.utils.data.DataLoader(valData, batch_size = batch, shuffle = False)
    
    mse = torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    
    fVloss= 0
    for e in range(epochs):
        loss = 0.0
        valLoss = 0.0
        for Xh, Xd, Xw, Y in trainLoader:
            Yp = model(Xh, Xd, Xw, A)
            lossT = mse(Yp, Y)
            optimizer.zero_grad()
            lossT.backward()
            optimizer.step()
            loss += lossT.item()
  
        with torch.set_grad_enabled(False):
                for Xh, Xd, Xw, Y in valLoader:
                    Yval = model(Xh, Xd, Xw, A)
                    lossV = mse(Yval, Y)
                    valLoss += lossV.item()
        
        epochLoss = loss / len(trainLoader)
        epochValLoss = loss/ len(valLoader)
        fVloss = epochValLoss
        print(f"Epoch: {e}, train_loss: {epochLoss}, validation_loss = {epochValLoss}")
    return model, fVloss

################################### MAIN ########################################
#train
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

device = setupCUDA()

astgcnlstm = ASTGCNLSTM(N, Th, Td, Tw, Tp)
countParameters(astgcnlstm)
astgcnlstm.to(device)

batch = 32
epochs = 400
astgcnlstm, fVloss = train(astgcnlstm, Xh, Xd, Xw, Y, A, N, device, batch, epochs)

fn = f"Astgcnnormlstm_Th{Th}_Td{Td}_Tw{Tw}_64_32_b{batch}_e{epochs}_vl{fVloss}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"
saveModel(astgcnlstm, fn, "models", "PM10", "ASTGCNLSTM")
########################

#test
testPath = "PM10/test"
N, A, coordsDf = buildGraphFromJson('coords.json')
X_test, F, T = getPMLevelsImputed(testPath, coordsDf, N)
X_test, scaler = normalize(X_test)
X_test = X_test.reshape(N, 1, T)

Xh, Xd, Xw, Yt = extractComponents(X_test, Tp, Th, Td, Tw, q)
#Xh = Xh.reshape(Xh.shape[0], N, Th)
Xd = torch.Tensor(Xd).to(device)
Xw = torch.Tensor(Xw).to(device)
#Yt =   Yt.reshape(Yt.shape[0], N, Tp)
Xh = torch.Tensor(Xh).to(device)
A = torch.Tensor(A).to(device)


total = Xh.shape[0]
batch  = 64
start = 0
end = batch
Yp = np.array([])

while (start <= total ):
    Xh_b = Xh[start:end]
    Xd_b = Xd[start:end]
    Xw_b = Xw[start:end]
    Yp_b = astgcnlstm(Xh_b,Xd_b,Xw_b, A).cpu().detach().numpy()
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
print(mse(Yt[:, 0, :], Yp[:, 0, :]))  #953

