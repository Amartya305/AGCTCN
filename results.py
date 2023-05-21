# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:59:07 2021

@author: Amartya Choudhury
"""

from utils import buildGraphFromJson, getPMLevelsImputed, extractComponents, countParameters, normalize, setupCUDA
import os
import numpy as np
from astgcn import ASTGCN
from gclstm import GCLSTM
from lstm import LSTM, createSequences
from convlstm import *
import torch
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from visuals import plotPredictions, boxplotMetrices, correlationPlot
from taylor import TaylorDiagram, plotTaylor
from models import ASTGCN_noTatt
import math
def cov(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    sub_x = x - mean_x
    sub_y = y - mean_y
    return np.sum(sub_x*sub_y) / x.shape[0]

def r2(X, Y):
    N = X.shape[0]
    R2 = 0.0
    for i in range(N):
        R2 += r2_score(X[i, :], Y[i, :])
    return R2 / N
        
def pearson(X, Y):
    N = X.shape[0]
    T = X.shape[1]
    rhoSum = 0.0
    for i in range(N):
        rhoxy = cov(X[i], Y[i]) / (np.std(X[i]) * np.std(Y[i]))
        rhoSum += rhoxy
    return rhoSum / N

def mseNoAvg(X, Y):
    N = X.shape[0]
    mseAll = np.zeros((N))
    for i in range(N):
        rmse = mse(X[i, :], Y[i, :], squared = False)
        mseAll[i] = rmse
    return mseAll

def nrmse(X, Y):
    N = X.shape[0]
    nrmse = 0
    for i in range(N):
        nrmse += math.sqrt(mse(X[i, :], Y[i, :], squared = True) / (np.mean(X[i, :]) * np.mean(X[i, :])))
    return nrmse / N 

def nrmseNoAvg(X, Y):
    N = X.shape[0]
    nrmse = np.zeros((N))
    for i in range(N):
        nrmse[i] = math.sqrt(mse(X[i, :], Y[i, :], squared = True) / (np.mean(X[i, :]) * np.mean(X[i, :])))
    return nrmse 

def pearsonNoAvg(X, Y):
    N = X.shape[0]
    T = X.shape[1]
    rhoXY = np.zeros((N))
    for i in range(N):
        rhoxy = cov(X[i], Y[i]) / (np.std(X[i]) * np.std(Y[i]))
        rhoXY[i] = rhoxy
    return rhoXY

def r2NoAvg(X, Y):
    N = X.shape[0]
    R2 = np.zeros((N))
    for i in range(N):
        R2[i] = r2_score(X[i, :], Y[i, :])
    return R2

def maeNoAvg(X, Y):
    N = X.shape[0]
    maeAll = np.zeros((N))
    for i in range(N):
        maes = mae(X[i, :], Y[i, :])
        maeAll[i] = maes
    return maeAll

def savePreds(Y, Yp, pol, model, coordsDf):
    if os.path.exists("Preds") == 0:
        os.mkdir("Preds")
    polpath = os.path.join("Preds", pol)
    if os.path.exists(polpath) == 0:
        os.mkdir(polpath)
    modelpath = os.path.join(polpath, model)
    if os.path.exists(modelpath) == 0:
        os.mkdir(modelpath)
    for idx, station in enumerate(coordsDf["Stations"]):
        filename = os.path.join(modelpath, station +".csv")
        preds = pd.DataFrame(np.column_stack((Y[idx,0,:], Yp[idx,0,:])), columns = ["True", "Pred"])
        preds.to_csv(filename)

def testAGCTCN(X_test, pol, N, A, T, averaged):
    Tp = 12
    Th = 24
    Td = 12
    Tw = 24
    q= 96
    astgcn =  ASTGCN_noTatt(N, Th, Td, Tw, Tp)
    device = setupCUDA()
    astgcn.to(device)
    path = "models/PM10/AGCTCN_ablation/" + "Agctcn_noTatt_Th24_Td12_Tw24_32_64_32_b32_e200_vl10878.421212332589_02_26_33_29-12-21.pt"
    astgcn.load_state_dict(torch.load(path))
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
    Y = Y.reshape(-1, 1)
    Yp = Yp.reshape(obs, N, 1, Tp)
    Y = Y.reshape(obs, N, 1, Tp)

    Yp = np.transpose(Yp, (1, 0, 2, 3))   # N x obs X 1 X Tp
    Yp = np.reshape(Yp, (Yp.shape[0], 1, Yp.shape[1] * Yp.shape[3]))
    Y = np.transpose(Y, (1, 0, 2, 3))  
    Y = np.reshape(Y, (Y.shape[0], 1, Y.shape[1] * Y.shape[3]))

    if averaged == True:
        rmse = mse(Y[:, 0, :], Yp[:, 0, :], squared = False)
        corr = pearson(Y[:, 0, :], Yp[:, 0, :])
        maes = mae(Y[:, 0, :], Yp[:, 0, :])
        r2s= r2(Y[:, 0, :], Yp[:, 0, :])
        nmse = nrmse(Y[:, 0, :], Yp[:, 0, :])
    else:
        rmse = mseNoAvg(Y[:, 0, :], Yp[:, 0, :])
        corr = pearsonNoAvg(Y[:, 0, :], Yp[:, 0, :])
        maes = maeNoAvg(Y[:, 0, :], Yp[:, 0, :])
        r2s  = r2NoAvg(Y[:, 0, :], Yp[:, 0, :])
        nmse = nrmseNoAvg(Y[:, 0, :], Yp[:, 0, :])
    return Y, Yp, rmse, corr, maes, r2s, nmse
    
    
def testGCLSTM(X_test, pol, N, A, T, averaged):
    Tp = 12
    Th = 24
    Td = 12
    Tw = 24
    q = 24 * 4
    device = setupCUDA()
    gclstm = GCLSTM(Th, Tp)
    gclstm.to(device)
    path = "models/PM10/GCLSTM/" + "GCLSTM_Th24_Tp12_hidden_32_layers_2_b32_e400_vl0.010422150230234755_18_13_18_24-07-21.pt"
    gclstm.load_state_dict(torch.load(path))
    
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
    
    if averaged == True:
        rmse = mse(Yt[:, 0, :], Yp[:, 0, :], squared = False)
        corr = pearson(Yt[:, 0, :], Yp[:, 0, :])
        maes = mae(Yt[:, 0, :], Yp[:, 0, :])
        r2s= r2(Yt[:, 0, :], Yp[:, 0, :])
        nmse = nrmse(Yt[:, 0, :], Yp[:, 0, :])
    else:
        rmse = mseNoAvg(Yt[:, 0, :], Yp[:, 0, :])
        corr = pearsonNoAvg(Yt[:, 0, :], Yp[:, 0, :])
        maes = maeNoAvg(Yt[:, 0, :], Yp[:, 0, :])
        r2s  = r2NoAvg(Yt[:, 0, :], Yp[:, 0, :])
        nmse = nrmseNoAvg(Yt[:, 0, :], Yp[:, 0, :])
    return Yt, Yp, rmse, corr, maes, r2s, nmse

def testConvLSTM(X_test, pol, N, A ,T, coordsDf, averaged):
    X_test, scaler = normalize(X_test)
    X_test = X_test.reshape(N, 1, T)
    X_grid, gridMap = buildGrid(X_test, N, T, coordsDf)
    Th = 24
    Tp = 12
    Xh, Y = split(X_grid, Th, Tp)

    
    path = "models/PM10/ConvLSTM/" + "ConvLSTM_Th24_Tp12_b8_e100_16_32_64_vl0.0004147971452520154_14_06_56_04-08-21.pt"
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).cuda()
    decoder = Decoder(convlstm_decoder_params[0], convlstm_decoder_params[1]).cuda()
    clstm = ED(encoder, decoder)
    clstm.load_state_dict(torch.load(path))
    dev = setupCUDA()
    clstm.to(dev)
    Yp = np.array([])
    Xh = torch.Tensor(Xh).to(dev)
    
    total = Xh.shape[0]
    batch  = 8
    start = 0
    end = batch
    while (start <= total ):
        Xh_b = Xh[start:end].reshape(Xh[start:end].shape[0], Xh.shape[1], Xh.shape[2], Xh.shape[3], Xh.shape[4] )
        Yp_b = clstm(Xh_b).cpu().detach().numpy()
        if (Yp.size == 0):
            Yp = Yp_b
        else:
            Yp = np.vstack((Yp, Yp_b))
        start += batch
        end += batch
        print(Yp.shape)
    Y = Y.reshape(Y.shape[0]*Y.shape[1], 1, 48, 48)
    Yp = Yp.reshape(Yp.shape[0]*Yp.shape[1], 1, 48, 48)
    T_pred = Y.shape[0]
    
    Y = revertGrid(Y, gridMap, N, T_pred)
    Yp = revertGrid(Yp, gridMap,  N, T_pred)
    
    Yp = Yp.reshape(-1, 1)
    Yp = scaler.inverse_transform(Yp)
    Y = Y.reshape(-1, 1)
    Y = scaler.inverse_transform(Y)
    
    Yp = Yp.reshape(N, 1, -1)
    Y = Y.reshape(N, 1, -1)
    if averaged == True:
        rmse = mse(Y[:, 0, :], Yp[:, 0, :], squared = False)
        corr = pearson(Y[:, 0, :], Yp[:, 0, :])
        maes = mae(Y[:, 0, :], Yp[:, 0, :])
        r2s= r2(Y[:, 0, :], Yp[:, 0, :])
        nmse = nrmse(Y[:, 0, :], Yp[:, 0, :])
    else:
        rmse = mseNoAvg(Y[:, 0, :], Yp[:, 0, :])
        corr = pearsonNoAvg(Y[:, 0, :], Yp[:, 0, :])
        maes = maeNoAvg(Y[:, 0, :], Yp[:, 0, :])
        r2s  = r2NoAvg(Y[:, 0, :], Yp[:, 0, :])
        nmse = nrmseNoAvg(Y[:, 0, :], Yp[:, 0, :])
    return Y, Yp, rmse, corr, maes, r2s, nmse

def testLSTM(X_test, pol, N, A, T, averaged):
    Td = 24
    Tp = 12
    device = setupCUDA()
    lstm = LSTM(1, 32, 2, 1, Tp)
    lstm.to(device)
    path = "models/PM10/LSTM/" + "Lstm_Td24_Tp12_32_2_b128_e400_vl0.023_00_36_02_12-07-21.pt"
    lstm.load_state_dict(torch.load(path))
    
    X_test, scaler = normalize(X_test)
    Xt, Yt = createSequences(X_test, Td, Tp, N, T)
    Xt = torch.Tensor(Xt).to(device)
    Xt = Xt.reshape(-1, Td, 1)
    
    
    Yp = lstm(Xt).cpu().detach().numpy()
    
    Yp = Yp.reshape(-1, 1)
    Yp = scaler.inverse_transform(Yp)
    Yt = Yt.reshape(-1, 1)
    Yt = scaler.inverse_transform(Yt)
    
    Yp = Yp.reshape(N, 1, -1)
    Yt = Yt.reshape(N, 1, -1)
    if averaged == True:
        rmse = mse(Yt[:, 0, :], Yp[:, 0, :], squared = False)
        corr = pearson(Yt[:, 0, :], Yp[:, 0, :])
        maes = mae(Yt[:, 0, :], Yp[:, 0, :])
        r2s= r2(Yt[:, 0, :], Yp[:, 0, :])
        nmse = nrmse(Yt[:, 0, :], Yp[:, 0, :])
    else:
        rmse = mseNoAvg(Yt[:, 0, :], Yp[:, 0, :])
        corr = pearsonNoAvg(Yt[:, 0, :], Yp[:, 0, :])
        maes = maeNoAvg(Yt[:, 0, :], Yp[:, 0, :])
        r2s  = r2NoAvg(Yt[:, 0, :], Yp[:, 0, :])
        nmse = nrmseNoAvg(Yt[:, 0, :], Yp[:, 0, :])
    return Yt, Yp, rmse, corr, maes, r2s, nmse

def testARIMA(X_test, pol):
    return

def testAll(averaged = True):
    N, A, coordsDf = buildGraphFromJson('coords.json')
    pollutants = ["PM10", "PM25"]
    RMSE = [dict(), dict()] 
    CORR = [dict(), dict()]
    MAE = [dict(), dict()] 
    R2 = [dict(), dict()]
    NMSE = [dict(), dict()]
    Ytrue = [dict(), dict()] 
    Ypred = [dict(), dict()]
    STD = [dict(), dict()]
    for idx, pol in enumerate(pollutants):
        testPath = os.path.join(pol,"test")
        X_test, F, T = getPMLevelsImputed(testPath, coordsDf, N)
        
        Y, Yp, rmse, corr, mae, r2s, nmse = testAGCTCN(X_test, pol, N, A, T, averaged)
        RMSE[idx]["AGCTCN"] = rmse
        CORR[idx]["AGCTCN"] = corr        
        MAE[idx]["AGCTCN"] = mae
        R2[idx]["AGCTCN"] = r2s 
        NMSE[idx]["AGCTCN"] = nmse 
        Ytrue[idx]["AGCTCN"] = Y
        Ypred[idx]["AGCTCN"] = Yp  
        savePreds(Y, Yp, pol, "AGCTCN", coordsDf)
    
        Y, Yp, rmse, corr, mae, r2s, nmse = testGCLSTM(X_test, pol, N, A, T, averaged)
        RMSE[idx]["GCLSTM"] = rmse
        CORR[idx]["GCLSTM"] = corr    
        MAE[idx]["GCLSTM"] = mae
        R2[idx]["GCLSTM"] = r2s
        NMSE[idx]["GCLSTM"] = nmse 
        Ytrue[idx]["GCLSTM"] = Y
        Ypred[idx]["GCLSTM"] = Yp 
        savePreds(Y, Yp, pol, "GCLSTM", coordsDf)

        Y, Yp, rmse, corr, mae,r2s, nmse = testConvLSTM(X_test, pol, N, A, T, coordsDf, averaged)
        RMSE[idx]["ConvLSTM"] = rmse
        CORR[idx]["ConvLSTM"] = corr
        MAE[idx]["ConvLSTM"] = mae
        R2[idx]["ConvLSTM"] = r2s
        NMSE[idx]["ConvLSTM"] = nmse 
        Ytrue[idx]["ConvLSTM"] = Y
        Ypred[idx]["ConvLSTM"] = Yp 
        savePreds(Y, Yp, pol, "ConvLSTM", coordsDf)
        
        Y, Yp, rmse, corr, mae, r2s, nmse = testLSTM(X_test, pol, N, A, T, averaged)
        RMSE[idx]["LSTM"] = rmse
        CORR[idx]["LSTM"] = corr
        MAE[idx]["LSTM"] = mae
        R2[idx]["LSTM"] = r2s
        NMSE[idx]["LSTM"] = nmse 
        Ytrue[idx]["LSTM"] = Y
        Ypred[idx]["LSTM"] = Yp 
        savePreds(Y, Yp, pol, "LSTM", coordsDf)
        #print(RMSE)
        #(CORR)
        
    return RMSE, CORR, MAE, R2, NMSE, Ytrue, Ypred

def saveStationWiseMetrics(metric, name):
    pd.DataFrame(metric[0]).to_csv("Preds/" + "PM10_" + name +".csv")
    pd.DataFrame(metric[1]).to_csv("Preds/" + "PM25_" + name +".csv")
    
### MAIN ###
'''
RMSE, CORR, MAE, R2, NMSE, Ytrue, Ypred = testAll()
pd.DataFrame(RMSE).to_csv("Preds/rmse.csv")
pd.DataFrame(CORR).to_csv("Preds/Correlation.csv")
pd.DataFrame(MAE).to_csv("Preds/mae.csv")
pd.DataFrame(R2).to_csv("Preds/r2_score.csv")
pd.DataFrame(NMSE).to_csv("Preds/nrmse.csv")
'''

RMSE, CORR, MAE, R2, NMSE, Ytrue, Ypred = testAll(0)
saveStationWiseMetrics(RMSE, "RMSE")
saveStationWiseMetrics(NMSE, "NRMSE")
saveStationWiseMetrics(CORR, "CORR")
saveStationWiseMetrics(MAE, "MAE")
saveStationWiseMetrics(R2, "R2")

boxplotMetrices(RMSE, NMSE, CORR, MAE, R2)
'''
correlationPlot(Ytrue, Ypred)

STD_obs = []
STD_pred = [dict(), dict()]
for i in range(2):
    stdobs = []
    stdpredAGCTCN = []
    stdpredGCLSTM = []
    stdpredConvLstm = []
    stdpredLstm = []
    for n in range(27):
        stdobs.append(np.std(Ytrue[i]["AGCTCN"][n]))
        stdpredAGCTCN.append(np.std(Ypred[i]["AGCTCN"][n]))
        stdpredGCLSTM.append(np.std(Ypred[i]["GCLSTM"][n]))
        stdpredConvLstm.append(np.std(Ypred[i]["ConvLSTM"][n]))
        stdpredLstm.append(np.std(Ypred[i]["LSTM"][n]))
    STD_pred[i]["AGCTCN"] = np.array(stdpredAGCTCN)
    STD_pred[i]["GCLSTM"] = np.array(stdpredGCLSTM)
    STD_pred[i]["ConvLstm"] = np.array(stdpredConvLstm)
    STD_pred[i]["LSTM"] = np.array(stdpredLstm)
    STD_obs.append(np.array(stdobs))

pollutants = ["$PM_{10}$", "$PM_{2.5}$"]
for i in range(2):
    stdobs_mean = np.mean(STD_obs[i])
    stdpred_mean = []
    corrpred_mean = []
    labels = []
    for model in STD_pred[i].keys():
        stdpred_mean.append(np.mean(STD_pred[i][model]))
        
    for model in CORR[i].keys():
        corrpred_mean.append(np.mean(CORR[i][model]))
        labels.append(model)
    plotTaylor(stdobs_mean, stdpred_mean, corrpred_mean, labels, pollutants[i])
'''    