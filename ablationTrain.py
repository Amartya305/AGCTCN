# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 21:41:38 2021

@author: amart
"""
from utils import buildGraphFromJson, getPMLevelsImputed, extractComponents, countParameters, normalize
from models import *
from sklearn.model_selection import train_test_split
from datetime import datetime
import cProfile
import pstats
import torch
import random
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler



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

    
def train(model, Xh, Xd, Xw, Y, A, N, device, batch=32, epochs=200):
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

    
def train2(model1, model2, model3, Xh, Xd, Xw, Y, A, N, device, batch=32, epochs=200):
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
    optimizer1=torch.optim.Adam(model1.parameters(),lr=0.01)
    optimizer2=torch.optim.Adam(model2.parameters(),lr=0.01)
    optimizer3=torch.optim.Adam(model3.parameters(),lr=0.01)
    
    fVloss1 = 0
    fVloss2 = 0
    fVloss3 = 0
    
    for e in range(epochs):
        loss = 0.0
        valLoss = 0.0
        for Xh, Xd, Xw, Y in trainLoader:
            Yp = model1(Xh, Xd, A)
            lossT = mse(Yp, Y)
            optimizer1.zero_grad()
            lossT.backward()
            optimizer1.step()
            loss += lossT.item()
        with torch.set_grad_enabled(False):
                for Xh, Xd, Xw, Y in valLoader:
                    Yval = model1(Xh, Xd, A)
                    lossV = mse(Yval, Y)
                    valLoss += lossV.item()
        
        epochLoss = loss / len(trainLoader)
        epochValLoss = loss/ len(valLoader)
        fVloss1 = epochValLoss
        print(f"Model1: Epoch: {e}, train_loss: {epochLoss}, validation_loss = {epochValLoss}")

    for e in range(epochs):
        loss = 0.0
        valLoss = 0.0
        for Xh, Xd, Xw, Y in trainLoader:
            Yp = model2(Xh, Xw, A)
            lossT = mse(Yp, Y)
            optimizer2.zero_grad()
            lossT.backward()
            optimizer2.step()
            loss += lossT.item()
        with torch.set_grad_enabled(False):
                for Xh, Xd, Xw, Y in valLoader:
                    Yval = model2(Xh, Xw, A)
                    lossV = mse(Yval, Y)
                    valLoss += lossV.item()
        
        epochLoss = loss / len(trainLoader)
        epochValLoss = loss/ len(valLoader)
        fVloss2 = epochValLoss
        print(f"Model2: Epoch: {e}, train_loss: {epochLoss}, validation_loss = {epochValLoss}")

    for e in range(epochs):
        loss = 0.0
        valLoss = 0.0
        for Xh, Xd, Xw, Y in trainLoader:
            Yp = model3(Xd, Xw, A)
            lossT = mse(Yp, Y)
            optimizer3.zero_grad()
            lossT.backward()
            optimizer3.step()
            loss += lossT.item()
        with torch.set_grad_enabled(False):
                for Xh, Xd, Xw, Y in valLoader:
                    Yval = model3(Xd, Xw, A)
                    lossV = mse(Yval, Y)
                    valLoss += lossV.item()
        
        epochLoss = loss / len(trainLoader)
        epochValLoss = loss/ len(valLoader)
        fVloss3 = epochValLoss
        print(f"Model3: Epoch: {e}, train_loss: {epochLoss}, validation_loss = {epochValLoss}")

    return model1, model2, model3, fVloss1, fVloss2, fVloss3

def saveModel(model, filename):
    modelDir  = "models"
    if (os.path.exists(modelDir) == 0):
       os.mkdir(modelDir) 
    polnDir  = os.path.join(modelDir, "PM10")
    if (os.path.exists(polnDir) == 0):
       os.mkdir(polnDir)   
    astgcnDir = os.path.join(polnDir, "AGCTCN_ablation")
    if (os.path.exists(astgcnDir) == 0):
        os.mkdir(astgcnDir)  
    torch.save(model.state_dict(), os.path.join(os.getcwd(), astgcnDir, filename))
    return

N, A, coordsDf = buildGraphFromJson('coords.json')
trainPath = "PM10/train"
X_train, F, T = getPMLevelsImputed(trainPath, coordsDf, N)

#X_train, scaler = normalize(X_train)
X_train = X_train.reshape(N, 1, T)
Tp = 12
Th = 24
Td = 12
Tw = 24
q = 24 * 4

#extract weekly, hourly and recent components
Xh, Xd, Xw, Y = extractComponents(X_train, Tp, Th, Td, Tw, q)
device = setupCUDA()
batch = 32
epochs = 200
'''
astgcn = ASTGCN_noSatt(N, Th, Td, Tw, Tp)
countParameters(astgcn)
astgcn.to(device)
astgcn, fVloss = train(astgcn, Xh, Xd, Xw, Y, A, N, device, batch, epochs)
fn = f"Agctcn_noSatt_Th{Th}_Td{Td}_Tw{Tw}_32_64_32_b{batch}_e{epochs}_vl{fVloss}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"
saveModel(astgcn, fn)

astgcn = ASTGCN_noTatt(N, Th, Td, Tw, Tp)
countParameters(astgcn)
astgcn.to(device)
astgcn.load_state_dict(torch.load("models/PM10/AGCTCN_ablation/" + "Agctcn_noTatt_Th24_Td12_Tw24_32_64_32_b32_e200_vl10878.421212332589_02_26_33_29-12-21.pt"))
astgcn, fVloss = train(astgcn, Xh, Xd, Xw, Y, A, N, device, batch, epochs)
fn = f"Agctcn_noTatt_Th{Th}_Td{Td}_Tw{Tw}_32_64_32_b{batch}_e400_vl{fVloss}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"
saveModel(astgcn, fn)

astgcn = ASTGCN_noGCN(N, Th, Td, Tw, Tp)
countParameters(astgcn)
astgcn.to(device)
astgcn, fVloss = train(astgcn, Xh, Xd, Xw, Y, A, N, device, batch, epochs)
fn = f"Agctcn_noGCN_Th{Th}_Td{Td}_Tw{Tw}_32_64_32_b{batch}_e{epochs}_vl{fVloss}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"
saveModel(astgcn, fn)

astgcn = ASTGCN_noTime(N, Th, Td, Tw, Tp)
countParameters(astgcn)
astgcn.to(device)
astgcn, fVloss = train(astgcn, Xh, Xd, Xw, Y, A, N, device, batch, epochs)
fn = f"Agctcn_noTime_Th{Th}_Td{Td}_Tw{Tw}_32_64_32_b{batch}_e{epochs}_vl{fVloss}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"
saveModel(astgcn, fn)
'''

astgcn1 = ASTGCN_noWeekly(N, Th, Td, Tp)
astgcn2 = ASTGCN_noDaily(N, Th, Tw, Tp)
astgcn3 = ASTGCN_noRecent(N, Td, Tw, Tp)
countParameters(astgcn1)
astgcn1.to(device)
astgcn2.to(device)
astgcn3.to(device)
astgcn1, astgcn2, astgcn3, fVloss1, fVloss2, fVloss3 = train2(astgcn1, astgcn2, astgcn3, Xh, Xd, Xw, Y, A, N, device, batch, epochs)
fn1 = f"Agctcn_noWeekly_Th{Th}_Td{Td}_Tw{Tw}_32_64_32_b{batch}_e{epochs}_vl{fVloss1}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"
saveModel(astgcn1, fn1)
fn2 = f"Agctcn_noDaily_Th{Th}_Td{Td}_Tw{Tw}_32_64_32_b{batch}_e{epochs}_vl{fVloss2}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"
saveModel(astgcn2, fn2)
fn3 = f"Agctcn_noRecent_Th{Th}_Td{Td}_Tw{Tw}_32_64_32_b{batch}_e{epochs}_vl{fVloss3}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"
saveModel(astgcn3, fn3)
