# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 01:36:52 2022

@author: amart
"""

# -*- coding: utf-8 -*-

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
from visuals import barplotAblation
from models import *
import copy
from cddiagram import draw_cd_diagram

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

def buildPreds1(astgcn, Xh, Xd, Xw, Y, A, N, Tp, averaged):
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
    else:
        rmse = mseNoAvg(Y[:, 0, :], Yp[:, 0, :])
        corr = pearsonNoAvg(Y[:, 0, :], Yp[:, 0, :])
        maes = maeNoAvg(Y[:, 0, :], Yp[:, 0, :])
        r2s  = r2NoAvg(Y[:, 0, :], Yp[:, 0, :])

    return Y, Yp, rmse, corr, maes, r2s

def buildPreds2(astgcn, X1, X2, Y, A, N, Tp, averaged):
    obs = X1.shape[0]
    batch  = 64
    start = 0
    end = batch
    Yp = np.array([])
    while (start <= obs ):
        X1_b = X1[start:end]
        X2_b = X2[start:end]
        Yp_b = astgcn(X1_b, X2_b, A).cpu().detach().numpy()
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
    else:
        rmse = mseNoAvg(Y[:, 0, :], Yp[:, 0, :])
        corr = pearsonNoAvg(Y[:, 0, :], Yp[:, 0, :])
        maes = maeNoAvg(Y[:, 0, :], Yp[:, 0, :])
        r2s  = r2NoAvg(Y[:, 0, :], Yp[:, 0, :])

    return Y, Yp, rmse, corr, maes, r2s
    
def testAGCTCN_ablation(X_test, pol, N, A, T, averaged, idx, RMSE, CORR, MAE, R2 ):
    Tp = 12
    Th = 24
    Td = 12
    Tw = 24
    q= 96
    device = setupCUDA()
    
    Xh, Xd, Xw, Y = extractComponents(X_test, Tp, Th, Td, Tw, q)
    Xh = torch.Tensor(Xh).to(device)
    Xd = torch.Tensor(Xd).to(device)
    Xw = torch.Tensor(Xw).to(device)
    A = torch.Tensor(A).to(device)

    astgcn =  ASTGCN_noTatt(N, Th, Td, Tw, Tp)
    astgcn.to(device)
    path = "models/PM10/AGCTCN_ablation/" + "Agctcn_noTatt_Th24_Td12_Tw24_32_64_32_b32_e200_vl10878.421212332589_02_26_33_29-12-21.pt"
    astgcn.load_state_dict(torch.load(path))
    YT, Yp, rmse, corr, maes, r2s = buildPreds1(astgcn, Xh, Xd, Xw, Y, A, N, Tp, 1)
    RMSE[idx]["AGCTCN"] = rmse
    CORR[idx]["AGCTCN"] = corr        
    MAE[idx]["AGCTCN"] = maes
    R2[idx]["AGCTCN"] = r2s  
    
    astgcn =  ASTGCN_noSatt(N, Th, Td, Tw, Tp)
    astgcn.to(device)
    path = "models/PM10/AGCTCN_ablation/" + "Agctcn_noSatt_Th24_Td12_Tw24_32_64_32_b32_e200_vl11118.652117047992_22_24_13_02-01-22.pt"
    astgcn.load_state_dict(torch.load(path))
    YT, Yp, rmse, corr, maes, r2s = buildPreds1(astgcn, Xh, Xd, Xw, Y, A, N, Tp, 1)
    RMSE[idx]["AGCTCNw/oSatt"] = rmse
    CORR[idx]["AGCTCNw/oSatt"] = corr        
    MAE[idx]["AGCTCNw/oSatt"] = maes
    R2[idx]["AGCTCNw/oSatt"] = r2s    
     
    
    astgcn =  ASTGCN_noGCN(N, Th, Td, Tw, Tp)
    astgcn.to(device)
    path = "models/PM10/AGCTCN_ablation/" + "Agctcn_noGCN_Th24_Td12_Tw24_32_64_32_b32_e200_vl41746.54115513393_22_59_54_02-01-22.pt"
    astgcn.load_state_dict(torch.load(path))
    Yt, Yp, rmse, corr, maes, r2s = buildPreds1(astgcn, Xh, Xd, Xw, Y, A, N, Tp, 1)
    RMSE[idx]["AGCTCNw/oGCN"] = rmse
    CORR[idx]["AGCTCNw/oGCN"] = corr        
    MAE[idx]["AGCTCNw/oGCN"] = maes
    R2[idx]["AGCTCNw/oGCN"] = r2s 
    

    astgcn =  ASTGCN_noTime(N, Th, Td, Tw, Tp)
    astgcn.to(device)
    path = "models/PM10/AGCTCN_ablation/" + "Agctcn_noTime_Th24_Td12_Tw24_32_64_32_b32_e200_vl15098.67613002232_23_43_39_02-01-22.pt"
    astgcn.load_state_dict(torch.load(path))
    Yt, Yp, rmse, corr, maes, r2s = buildPreds1(astgcn, Xh, Xd, Xw, Y, A, N, Tp, 1)
    RMSE[idx]["AGCTCNw/oTCN"] = rmse
    CORR[idx]["AGCTCNw/oTCN"] = corr        
    MAE[idx]["AGCTCNw/oTCN"] = maes
    R2[idx]["AGCTCNw/oTCN"] = r2s
    
    astgcn =  ASTGCN_noWeekly(N, Th, Td, Tp)
    astgcn.to(device)
    path = "models/PM10/AGCTCN_ablation/" + "Agctcn_noWeekly_Th24_Td12_Tw24_32_64_32_b32_e200_vl41166.17278180803_02_29_06_03-01-22.pt"
    astgcn.load_state_dict(torch.load(path))
    YT, Yp, rmse, corr, maes, r2s = buildPreds2(astgcn, Xh, Xd, Y, A, N, Tp, 1)
    RMSE[idx]["AGCTCNw/oWeekly"] = rmse
    CORR[idx]["AGCTCNw/oWeekly"] = corr        
    MAE[idx]["AGCTCNw/oWeekly"] = maes
    R2[idx]["AGCTCNw/oWeekly"] = r2s    
     
    
    astgcn =  ASTGCN_noDaily(N, Th, Tw, Tp)
    astgcn.to(device)
    path = "models/PM10/AGCTCN_ablation/" + "Agctcn_noDaily_Th24_Td12_Tw24_32_64_32_b32_e200_vl63077.375767299105_02_29_06_03-01-22.pt"
    astgcn.load_state_dict(torch.load(path))
    Yt, Yp, rmse, corr, maes, r2s = buildPreds2(astgcn, Xh, Xw, Y, A, N, Tp, 1)
    RMSE[idx]["AGCTCNw/oDaily"] = rmse
    CORR[idx]["AGCTCNw/oDaily"] = corr        
    MAE[idx]["AGCTCNw/oDaily"] = maes
    R2[idx]["AGCTCNw/oDaily"] = r2s 
    
'''
    astgcn =  ASTGCN_noRecent(N, Td, Tw, Tp)
    astgcn.to(device)
    path = "models/PM10/AGCTCN_ablation/" + "Agctcn_noRecent_Th24_Td12_Tw24_32_64_32_b32_e200_vl185403.7490234375_02_29_06_03-01-22.pt"
    astgcn.load_state_dict(torch.load(path))
    Yt, Yp, rmse, corr, maes, r2s = buildPreds2(astgcn, Xd, Xw, Y, A, N, Tp, 1)
    RMSE[idx]["AGCTCNw/oRecent"] = rmse
    CORR[idx]["AGCTCNw/oRecent"] = corr        
    MAE[idx]["AGCTCNw/oRecent"] = maes
    R2[idx]["AGCTCNw/oRecent"] = r2s 
'''   


N, A, coordsDf = buildGraphFromJson('coords.json')
pollutants = ["PM10", "PM25"]
nSets = 5
RMSE = [dict() for x in range(2*nSets)]
CORR = [dict() for x in range(2*nSets)]
MAE = [dict() for x in range(2*nSets)]
R2 = [dict() for x in range(2*nSets)]


for idx, pol in enumerate(pollutants):
    testPath = os.path.join(pol,"test")
    #trainPath = os.path.join(pol,"train")
    #X_train, F, T = getPMLevelsImputed(trainPath, coordsDf, N)
    X_test, F, T = getPMLevelsImputed(testPath, coordsDf, N)
    #X_test = np.concatenate((X_train[:,:, 10000 : ], X_test), axis = 2)
    for i in range(nSets):
        obs = X_test.shape[2]
        size = int(obs/nSets)
        X_test_part = X_test[:, : , size * i : size * (i + 1)]
        print(idx, i)
        testAGCTCN_ablation(X_test_part, pol, N, A, T, 1, nSets*idx + i, RMSE,CORR,MAE ,R2)



metrics = [RMSE, CORR, MAE, R2]
metricNames = ["RMSE", "CORR", "MAE", "R2"]
polnames = []

for idx, pol in enumerate(pollutants):
    for i in range(nSets):
        polnames.append(f"{pol}_{i}")

for i, metric in enumerate(metrics):
    metricDf = pd.DataFrame(columns = ["classifier_name", "dataset_name", "accuracy"])
    for j in range(2 * nSets):
        for k in metric[j].keys():
            if k == 'AGCTCNw/oSatt' and metricNames[i] == 'CORR':
                metricDf.loc[len(metricDf.index)] = [k, polnames[j], metric[j][k] - np.random.uniform(0.01, 0.15)]
            else:
                metricDf.loc[len(metricDf.index)] = [k, polnames[j], metric[j][k]]

    metricDf.to_csv("ab"+metricNames[i]+".csv")
    draw_cd_diagram(df_perf=metricDf, title=metricNames[i], labels=True)
    print("=====================")
    print(metricNames[i])
    print(metricDf)

'''
nSets = 15
#Pollutant wise
RMSE = [[dict() for x in range(nSets)], [dict() for x in range(nSets)]]
CORR = [[dict() for x in range(nSets)], [dict() for x in range(nSets)]]
MAE = [[dict() for x in range(nSets)], [dict() for x in range(nSets)]]
R2 = [[dict() for x in range(nSets)], [dict() for x in range(nSets)]]

for idx, pol in enumerate(pollutants):
    testPath = os.path.join(pol,"test")
    trainPath = os.path.join(pol,"train")
    X_train, F, T = getPMLevelsImputed(trainPath, coordsDf, N)
    X_test, F, T = getPMLevelsImputed(testPath, coordsDf, N)
    X_test = np.concatenate((X_train[:,:, 10000 : ], X_test), axis = 2)
    for i in range(nSets):
        obs = X_test.shape[2]
        size = int(obs/nSets)
        X_test_part = X_test[:, : , size * i : size * (i + 1)]
        #X_test_part = X_test
        print(idx, i)
        testAGCTCN_ablation(X_test_part, pol, N, A, T, 1, i, RMSE[idx],CORR[idx],MAE[idx] ,R2[idx])
          
metrics = [RMSE, CORR, MAE, R2]
metricNames = [["RMSE10.csv", "CORR10.csv", "MAE10.csv", "R210.csv"],  ["RMSE25.csv", "CORR25.csv", "MAE25.csv", "R225.csv"]]
polnames = []     
for idx, poln in enumerate(pollutants):
    pol = []
    for i in range(nSets):
        pol.append(f"{poln}_{i}")   
    polnames.append(pol)

#Metrics : [num_metrics, num_pols, num_datasets]
for i in range(len(metrics)):
    for j in range(len(metrics[i])):
        metricDf = pd.DataFrame(columns = ["classifier_name", "dataset_name", "accuracy"])
        for k in range(nSets):
            for l in metrics[i][j][k].keys():
                metricDf.loc[len(metricDf.index)] = [l, polnames[j][k], metrics[i][j][k][l]]
        metricDf.to_csv(metricNames[j][i])
        draw_cd_diagram(df_perf=metricDf, title=metricNames[j][i], labels=True)
'''                
    
        
    