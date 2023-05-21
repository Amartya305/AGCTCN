# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 20:20:23 2021

@author: amart
"""
from utils import buildGraphFromJson, getPMLevelsImputed, extractComponents, countParameters, normalize
from astgcn import ASTGCN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from datetime import datetime
from visuals import plotPredictions
from models import ASTGCN_noTatt, ASTGCN_noGCN, ASTGCN_noTime
import torch
import random
import numpy as np
import os

def setupCUDA():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.empty_cache()
    np.random.seed(1)
    return device


def test(model, X, Th, Td, Tp, q, nTrain, device, path = None):

    Xh_t, Xd_t, Xw_t, Y_t = extractComponents(X, Tp, Th, Td, Tw, q)
    
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

N, A, coordsDf = buildGraphFromJson('coords.json')

Tp = 12
Th = 24
Td = 12
Tw = 24
q= 96

astgcn =  ASTGCN_noTime(N, Th, Td, Tw, Tp)
device = setupCUDA()
astgcn.to(device)

path = "models/PM10/AGCTCN_ablation/" + "Agctcn_noTime_Th24_Td12_Tw24_32_64_32_b32_e200_vl13339.671360560826_03_07_43_29-12-21.pt"
astgcn.load_state_dict(torch.load(path))

testPath = "PM10/test"
X_test, F, T = getPMLevelsImputed(testPath, coordsDf, N)
#X_test, scaler = normalize(X_test)
#X_test = X_test.reshape(N, 1, T)

Xh, Xd, Xw, Y = extractComponents(X_test, Tp, Th, Td, Tw, q)

Xh = torch.Tensor(Xh).to(device)
Xd = torch.Tensor(Xd).to(device)
Xw = torch.Tensor(Xw).to(device)
A = torch.Tensor(A).to(device)

obs = Xh.shape[0]
batch  = 64
start = 0
end = batch
Yp = np.array([])

while (start <= obs ):
    Xh_b = Xh[start:end]
    Xd_b = Xd[start:end]
    Xw_b = Xw[start:end]
    Yp_b = astgcn(Xh_b, Xd_b, Xw_b, A).cpu().detach().numpy()
    if (Yp.size == 0):
        Yp = Yp_b
    else:
        Yp = np.vstack((Yp, Yp_b))
    start += batch
    end += batch

Yp = Yp.reshape(-1, 1)
#Yp = scaler.inverse_transform(Yp)

Y = Y.reshape(-1, 1)
#Y = scaler.inverse_transform(Y)

Yp = Yp.reshape(obs, N, 1, Tp)
Y = Y.reshape(obs, N, 1, Tp)

Yp = np.transpose(Yp, (1, 0, 2, 3))   # N x obs X 1 X Tp
Yp = np.reshape(Yp, (Yp.shape[0], 1, Yp.shape[1] * Yp.shape[3]))

Y = np.transpose(Y, (1, 0, 2, 3))  
Y = np.reshape(Y, (Y.shape[0], 1, Y.shape[1] * Y.shape[3]))
plotPredictions(Y[:, :, 0:500], Yp[:, :, 0:500], coordsDf, N)
mse(Y[:, 0, :], Yp[:, 0, :])