# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import torch
from graph_utils import *
from utils import *
from models import *
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import cProfile
import pstats
from visuals import plotPredictions

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)



coordsDf = parseJson("coords.json", ["Stations", "Longitude", "Latitude"])
patchCoords(coordsDf)
coordsDf = coordsDf.drop(labels = [5,9,17,19,23,25,28,29,30])
coordsDf = coordsDf.reset_index(drop = True)
N, A = buildGraphFromCoords(coordsDf, ["Latitude", "Longitude"])


polnAll = []
for station in coordsDf["Stations"]:
    path = "Delhi/" + station +".csv"
    polnDf = pd.read_csv(path)
    polnAll.append(polnDf)

for poln in polnAll:
    linearInterpolateMissing(poln, ["PM10", "PM2.5"])

polnAll[4] = polnAll[4].loc[0:35232]

F = len(polnAll[0].columns) - 2
T = len(polnAll[0])

X_all = np.zeros(shape = (N, F, T))
for n in range(N):
    for f in range(F):
        df = polnAll[n]
        X_all[n][f] = np.array(df[df.columns[f + 2]])

Tp = 12
Th = 24
Td = 12
Tw = 24

trainSize = 34233
X_train = X_all[:, :, 0 : trainSize]
X_test = X_all[:, :, trainSize : 35232]

'''
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
'''

Xh, Xd, Xw, Y = extractComponents(X_train,
                                     Tp,
                                     Th,
                                     Td,
                                     Tw,
                                     q = 4 * 24 
                                     )

PM10idx = 0
PM25idx = 1

idx = PM10idx

Xh = Xh[:, :, idx, :]
Xh = Xh.reshape((Xh.shape[0], Xh.shape[1], 1, Xh.shape[2] ))


Xd = Xd[:, :, idx, :]
Xd= Xd.reshape((Xd.shape[0], Xd.shape[1], 1, Xd.shape[2] ))

Xw = Xw[:, :, idx, :]
Xw = Xw.reshape((Xw.shape[0], Xw.shape[1], 1, Xw.shape[2] ))

Y = Y[:, :, idx, :]
Y = Y.reshape((Y.shape[0], Y.shape[1], 1, Y.shape[2] ))

Xh = torch.Tensor(Xh)
Xd = torch.Tensor(Xd)
Xw = torch.Tensor(Xw)
Y = torch.Tensor(Y)
A = torch.Tensor(A)
A = A.to(device)

#train
class  PolnData(torch.utils.data.Dataset):
    def __init__(self, indices):
        self.indices = indices
        self.Xh = Xh[indices].to(device)
        self.Xd = Xd[indices].to(device)
        self.Xw = Xw[indices].to(device)
        self.Y = Y[indices].to(device)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.Xh[index], self.Xd[index], self.Xw[index], self.Y[index]

'''
batchSize = 32
indices = np.arange(Xh.shape[0])
idxTrain, idxVal = train_test_split(indices, test_size = 0.2, random_state = 42)
trainData = PolnData(idxTrain)
valData = PolnData(idxVal)
trainLoader = torch.utils.data.DataLoader(trainData, batch_size = batchSize, shuffle = True)
valLoader = torch.utils.data.DataLoader(valData, batch_size = batchSize, shuffle = False)

model = ASTGCN(N, Th, Td, Tw, Tp, A)
countParameters(model)
model.to(device)
mse = torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
epochs = 400

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


torch.save(model.state_dict(), "ASTGCN_complete_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y") + "_PM10_400.pt")

def train1():
    for e in range(1):
        loss = 0.0
        valLoss = 0.0
        for Xh, Xd, Xw, Y in trainLoader:
            Yp = model(Xh, Xd, Xw)
            lossT = mse(Yp, Y)
            optimizer.zero_grad()
            lossT.backward()
            optimizer.step()
            loss += lossT.item()
            print(lossT.item())
            break


profile = cProfile.Profile()
profile.runcall(train1)
ps = pstats.Stats(profile)
ps.print_stats().sort_stats('tottime')

X = torch.ones((32, 27, 24), device = 'cuda:0')
gcn = ChebConv(24, 64, 3).to('cuda:0')
Ab = A.repeat(32, 1, 1)
b, i , e = getGCNParams(Ab)
Y = gcn(X, i, e, b)
'''

#test

model = ASTGCNlight(N, Th, Td, Tw, Tp, A).to(device)
model.load_state_dict(torch.load('ASTGCN_complete_07_02_07_14-05-21_PM10_400.pt')) 
Xh_t, Xd_t, Xw_t, Y_t = extractComponents(X_all, Tp, Th, Td, Tw, q = 96, t0 = trainSize - 1)

Xh_t = Xh_t[:, :, idx, :]
Xh_t = Xh_t.reshape((Xh_t.shape[0], Xh_t.shape[1], 1, Xh_t.shape[2] ))


Xd_t = Xd_t[:, :, idx, :]
Xd_t = Xd_t.reshape((Xd_t.shape[0], Xd_t.shape[1], 1, Xd_t.shape[2] ))

Xw_t = Xw_t[:, :, idx, :]
Xw_t = Xw_t.reshape((Xw_t.shape[0], Xw_t.shape[1], 1, Xw_t.shape[2] ))

Y_t = Y_t[:, :, idx, :]
Y_t = Y_t.reshape((Y_t.shape[0], Y_t.shape[1], 1, Y_t.shape[2] ))

Xh_t = torch.Tensor(Xh_t).to(device)
Xd_t = torch.Tensor(Xd_t).to(device)
Xw_t = torch.Tensor(Xw_t).to(device)

Y_pred = model(Xh_t, Xd_t, Xw_t).cpu().detach().numpy()

Y_pred = np.transpose(Y_pred, (1, 0, 2, 3))  # N x obs X 1 X Tp
Y_pred =  np.reshape(Y_pred, (Y_pred.shape[0], 1, Y_pred.shape[1] * Y_pred.shape[3]))

Y_t = np.transpose(Y_t, (1, 0, 2, 3))  # N x obs X 1 X Tp
Y_t =  np.reshape(Y_t, (Y_t.shape[0], 1, Y_t.shape[1] * Y_t.shape[3]))

plotPredictions(Y_t, Y_pred, coordsDf, N)
