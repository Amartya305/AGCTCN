

# -*- coding: utf-8 -*-
"""
Created on Wed May  5 03:19:40 2021

@author: Amartya Choudhury
"""

import pandas as pd
import json
import numpy as np
from pandas.api.types import is_numeric_dtype
from prettytable import PrettyTable
from graph_utils import buildGraphFromCoords
from statsmodels.tsa.stattools import acf
from visuals import *
import torch
import random
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
def countParameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

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

def normalize(X):
    N = X.shape[0]
    T = X.shape[2]
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X.reshape(-1, 1))
    return X_norm.reshape(N, T), scaler

#This api parses json file in format {{index->{colValue1, colValue2, ...}}, ...}
def parseJson(fileName, colNames = None):
    with open(fileName, "r") as file:
        jsonStr = json.load(file)
        jsonDict = eval(jsonStr)
        
        indices = list(jsonDict.keys())
        nCols= len(jsonDict[indices[0]])
        
        assert nCols == len(colNames) -1 , "Number of columns obtained\
            from json file does not match the number of columns passed\
                as argument"
        
        columns = []
        columns.append(indices)
        for nCol in range(nCols):
            col = list(map(lambda x : jsonDict[x][nCol], indices))
            columns.append(col)
        
        data = {}
        for idx, name in enumerate(colNames):
            data[name] = columns[idx]
        
        return pd.DataFrame(data)
    
def dropNegativeRows(df):
    for col in df.columns.tolist():
        if (is_numeric_dtype(df[col])):
            df = df[df[col] > 0]
    return df.reset_index(drop = True)

def linearInterpolateMissing(df, colnames, identifier = "None"):
    for colname in colnames:
        df[colname].replace(identifier, np.nan, inplace = True)
        df[colname] = df[colname].astype("float")
        df[colname].interpolate('linear', inplace = True)
    return

def replaceNans(df, colnames, identifier = "None"):
    for colname in colnames:
        df[colname].replace(identifier, np.nan, inplace = True)
        df[colname] = df[colname].astype("float")
    return
 
def extractComponents(X, Tp, Th, Td, Tw, q, t0 = None):
    assert len(X.shape) == 3, "Expected vector of dim N X F X T"
    T = X.shape[2]
    assert Th % Tp == 0, "Recent Period length must be integral multiple of Tp"
    assert Td % Tp == 0, "Daily Period length must be integral multiple of Tp"
    assert Tw % Tp == 0, "Weekly Period length must be integral multiple of Tp"
    # t0 + 1 is the index of the 1st observation to be forecasted
    if (t0 == None):
        t0 = int(7 *(Tw / Tp)*q - 1)

    X_T = np.transpose(X, axes = (2, 0, 1)) # transpose to T X N X F
    Y = []
    Xw = []
    Xd = []
    Xh = []
    ti = t0 + 1
    while (ti + Tp <= T):
        Y_i = X_T[ti : ti + Tp]
        Xw_i = np.array([])
        Xd_i = np.array([])
        Xh_i = X_T[ti - Th: ti]
        
        #extract weekly features
        w = int(Tw / Tp)
        while (w > 0):
            if (Xw_i.size == 0):
                Xw_i = X_T[ti-w*7*q : ti-w*7*q+Tp]
            else:
                Xw_i = np.vstack((Xw_i, X_T[ti-w*7*q : ti-w*7*q+Tp]))
            w = w - 1           
        
        #extract daily features
        d = int(Td / Tp)
        while (d > 0):
            if (Xd_i.size == 0):
                Xd_i = X_T[ti-d*q : ti-d*q+Tp]
            else:
                Xd_i = np.vstack((Xd_i, X_T[ti-d*q : ti-d*q+Tp]))
            d = d - 1          
          
        Y.append(Y_i)
        Xw.append(Xw_i)
        Xd.append(Xd_i)
        Xh.append(Xh_i)
        ti = ti + Tp
   
    Y = np.asarray(Y)
    Xh = np.asarray(Xh)
    Xd = np.asarray(Xd)
    Xw = np.asarray(Xw)
    
    # n_obs X T X N X F  => n_obs X N X F X T
    Y = np.transpose(Y, axes = (0, 2, 3, 1))
    Xh = np.transpose(Xh, axes = (0, 2, 3, 1))
    Xd = np.transpose(Xd, axes = (0, 2, 3, 1))
    Xw = np.transpose(Xw, axes = (0, 2, 3, 1))
    
    return Xh, Xd, Xw, Y

def patchCoords(coords):
    coords.loc[1, "Longitude"] = 77.266902
    coords.loc[1, "Latitude"] = 28.499943   
    coords.loc[2, "Longitude"] = 77.206254
    coords.loc[2, "Latitude"] = 28.556882
    coords.loc[5, "Longitude"] = 77.214974
    coords.loc[5, "Latitude"] = 28.554001    
    coords.loc[14, "Longitude"] = 77.116008
    coords.loc[14, "Latitude"] = 28.746317
    coords.loc[17, "Longitude"] = 77.316896
    coords.loc[17, "Latitude"] = 28.670250
    coords.loc[18, "Longitude"] = 77.160101
    coords.loc[18, "Latitude"] = 28.699200
    coords.loc[20, "Longitude"] = 77.158081
    coords.loc[20, "Latitude"] = 28.639346
    coords.loc[21, "Longitude"] = 77.237335
    coords.loc[21, "Latitude"] = 28.612753 
    coords.loc[26, "Longitude"] = 77.274249
    coords.loc[26, "Latitude"] = 28.550955
    coords.loc[31, "Longitude"] = 77.242972
    coords.loc[31, "Latitude"] = 28.627707
    coords.loc[32, "Longitude"] = 77.209157
    coords.loc[32, "Latitude"] = 28.688535
    coords.loc[35, "Longitude"] = 77.272381
    coords.loc[35, "Latitude"] = 28.522946
   
def buildGraphFromJson(fileName, threshold = 0.5):
    #build adj matrix
    coordsDf = parseJson(fileName, ["Stations", "Longitude", "Latitude"])
    patchCoords(coordsDf)
    coordsDf = coordsDf.drop(labels = [5,9,17,19,23,25,28,29,30])  #drop stations with little data
    coordsDf = coordsDf.reset_index(drop = True)
    N, A = buildGraphFromCoords(coordsDf, ["Latitude", "Longitude"], threshold)
    return N, A, coordsDf

def getPMLevelsAllStations(folderName, coordsDf, N):
    polnAll = []
    for station in coordsDf["Stations"]:
        path = folderName + "/" + station +".csv"
        polnDf = pd.read_csv(path)
        polnAll.append(polnDf)

    for poln in polnAll:
        replaceNans(poln, ["PM10", "PM2.5"])

    #trim time series to 35232
    polnAll[4] = polnAll[4].loc[0:35232]                   

    F = len(polnAll[0].columns) - 2
    T = len(polnAll[0])

    #Build pollution level array
    X_all = np.zeros(shape = (N, F, T))
    for n in range(N):
        df = polnAll[n]
        for f in range(F):
            X_all[n][f] = np.array(df[df.columns[f + 2]])
    return X_all, F, T

def getPMLevelsImputed(folderName, coordsDf, N):
    polnAll = []
    for station in coordsDf["Stations"]:
        path = folderName + "/" + station +".csv"
        polnDf = pd.read_csv(path)
        polnAll.append(polnDf)

    F = len(polnAll[0].columns) -1
    T = len(polnAll[0])

    #Build pollution level array
    X_all = np.zeros(shape = (N, F, T))
    for n in range(N):
        df = polnAll[n]
        for f in range(F):
            X_all[n][f] = np.array(df[df.columns[f + 1]])
    return X_all, F, T

def trainTestSplit(X, nTrain, T):
    X_train = X[:, :, 0 : nTrain]
    X_test = X[:, :, nTrain : T]
    return X_train, X_test

def getSingleFeature(X, idx):
    X = X[:, :, idx, :]
    X = X.reshape((X.shape[0], X.shape[1], 1, X.shape[2] ))
    return X
                    
def buildAndPlotAcf(X):
    pm10 = X[:, 0, :]
    pm25 = X[:, 1, :]    

    pm10Avg = np.nanmean(pm10, axis = 0)
    pm25Avg = np.nanmean(pm25, axis = 0)    
    pm10Acf = acf(pm10Avg, nlags = 500)
    pm25Acf = acf(pm25Avg, nlags = 500) 

    plotAcf(pm10Avg, pm25Avg)

    return pm10Acf, pm25Acf

def getCovariogram(Z, lag = 0):
    assert len(Z.shape) == 2, "Expected N X T matrix"
    N = Z.shape[0]
    T = Z.shape[1]
    u = np.nanmean(Z, axis = 1)
    std = np.nanstd(Z, axis = 1)
    S = np.zeros((N, N))
    for j in range(lag, T):
        Zj = Z[:, j]
        Zlag = Z[:, j - lag]
        Zjnorm = np.reshape(Zj -u, (N, 1))
        Zlagnorm = np.reshape(Zlag -u, (1, N))
        C = np.matmul(Zjnorm, Zlagnorm)
        S += C / (T - lag)
    for i in range(N):
        for j in range(N):
            S[i][j] /= std[i]*std[j]
    return S

def saveModel(model, fileName, modelTopDir, polName, modelName):
    if (os.path.exists(modelTopDir) == 0):
        os.mkdir(modelTopDir) 

    polnDir  = os.path.join(modelTopDir, polName)
    if (os.path.exists(polnDir) == 0):
        os.mkdir(polnDir) 

    modelDir = os.path.join(polnDir, modelName)
    if (os.path.exists(modelDir) == 0):
        os.mkdir(modelDir)

    torch.save(model.state_dict(), os.path.join(os.getcwd(), modelDir, fileName))
    return


                                