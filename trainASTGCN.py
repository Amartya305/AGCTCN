# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:19:37 2021

@author: amart
"""
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
    fVlossBest = 20000
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
        epochValLoss = valLoss/ len(valLoader)
        fVloss = epochValLoss
        print(f"Epoch: {e}, train_loss: {epochLoss}, validation_loss = {epochValLoss}")
        if fVloss < fVlossBest:
            fVlossBest = fVloss
            fn = f"AGCTCN_checkpoint_{fVlossBest}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"
            torch.save(model.state_dict(), os.path.join(os.getcwd(), os.path.join("models", "PM10", "AGCTCN"), fn))
            print("=====MODEL CHECKPOINT=====")
    return model, fVloss

################################### MAIN ########################################

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

astgcn = ASTGCN(N, Th, Td, Tw, Tp)
countParameters(astgcn)
astgcn.to(device)
#path = "models/PM10/ASTGCN/" + "Astgcntcnnorm_Th24_Td12_Tw24_64_32_b32_e600_vl0.008802909670131547_20_42_05_16-10-21.pt"
#astgcn.load_state_dict(torch.load(path))
batch = 32
epochs = 200
astgcn, fVloss = train(astgcn, Xh, Xd, Xw, Y, A, N, device, batch, epochs)

'''
profile = cProfile.Profile()
profile.runcall(train, astgcn, Xh, Xd, Xw, Y, A, N, device, batch, epochs)
ps = pstats.Stats(profile)
ps.strip_dirs().sort_stats(-1).print_stats()
'''
fn = f"Astgcn_Th{Th}_Td{Td}_Tw{Tw}_32_64_32_b{batch}_e{epochs}_vl{fVloss}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"

modelDir  = "models"
if (os.path.exists(modelDir) == 0):
   os.mkdir(modelDir) 

polnDir  = os.path.join(modelDir, "PM10")
if (os.path.exists(polnDir) == 0):
   os.mkdir(polnDir) 


astgcnDir = os.path.join(polnDir, "ASTGCN")
if (os.path.exists(astgcnDir) == 0):
    os.mkdir(astgcnDir)

torch.save(astgcn.state_dict(), os.path.join(os.getcwd(), astgcnDir, fn))
