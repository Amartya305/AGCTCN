# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 03:48:49 2021

@author: Amartya Choudhury
"""

from utils import buildGraphFromJson, getPMLevelsAllStations, getPMLevelsImputed
import os
import numpy as np
import pandas as pd
from pykrige.ok3d import OrdinaryKriging3D
import random
def countMaxConsecutiveNaNs(X, N, F, T):
    maxCount = 0
    for i in range(N):
        for j in range(F):
            nans = np.isnan(X[i][j]).nonzero()[0]
            if (nans.size == 0):
                continue
            curIdx = nans[0]
            count = 1
            for k in range(1, nans.shape[0]):
                if (nans[k] == curIdx + 1):
                    curIdx += 1
                    count += 1
                else:
                    #maxCount = count if count > maxCount else maxCount
                    if (count > maxCount):
                        maxCount = count
                        print(f"{i},{j},{nans[k]}:{maxCount}")
                    curIdx = nans[k]
                    count = 1
    return maxCount

def showConsecutiveNaNRanges(X, N, F, T):
    totalNaNs = 0
    for i in range(N):
        for j in range(F):
            nans = np.isnan(X[i][j]).nonzero()[0]
            if (nans.size == 0):
                continue
            totalNaNs += nans.shape[0]
            curIdx = nans[0]
            count = 1
            for k in range(1, nans.shape[0]):
                if (nans[k] == curIdx + 1):
                    curIdx += 1
                    count += 1
                else:
                    #maxCount = count if count > maxCount else maxCount
                    print(f"{i},{j},{nans[k - count]} :{nans[k - 1]}")
                    curIdx = nans[k]
                    count = 1
    print(f"Total: {totalNaNs}")
    return

def imputeMissingVals(X, N, F, T, A):
    for n in range(N):
        for f in range(F):
            for t in range(T):
                if (np.isnan(X[n][f][t])):
                    W = 0
                    WX = 0
                    for j in range(N):
                        w = A[n][j]
                        x = X[j][f][t]
                        if (np.isnan(x)):
                            continue
                        WX += w*x
                        W += w
                    if (W == 0):
                        continue
                    X[n][f][t] = (WX) / W
    return

def savePMLevelsAllStationImputed(X, N, F, T, coordsDf, folder):
    dir_ = os.path.join(os.getcwd(), folder)
    assert os.path.exists(dir_) == 0, "Folder already exists."
    os.mkdir(dir_)
    for idx, st in enumerate(coordsDf["Stations"]):
        filepath = os.path.join(dir_, st + ".csv")
        df = pd.DataFrame(X_all[idx].T, columns = ["PM10", "PM2.5"])
        df.to_csv(filepath)
    return

def setupKriging(coordsDf, batch = 12):
    long = np.array(coordsDf["Longitude"])
    lat = np.array(coordsDf["Latitude"])
    long = 100 * (long - 77)
    lat = 100 * (lat - 28)
    
    long = long.astype('int32')
    lat = lat.astype('int32')
    
    x = long
    y = lat
    xmin = x.min()
    ymin = y.min()
    xmax = x.max()
    ymax = y.max()
    
    x = np.repeat(x, batch)
    y = np.repeat(y, batch)
    z = np.arange(0 , batch)
    z = np.reshape(z, (1, batch))
    z = np.repeat(z, N, axis = 0)
    z = z.flatten()

    gridx = np.arange(xmin, xmax + 1).astype('float64')
    gridy = np.arange(ymin, ymax + 1).astype('float64')
    gridz = np.arange(0, batch).astype('float64')           
    return long, lat, x, y, z, xmin, ymin, xmax, ymax,  gridx, gridy, gridz

N, A, coordsDf = buildGraphFromJson('coords.json', threshold = 0)
X_all, F, T = getPMLevelsAllStations('Delhi', coordsDf, N)


nans = []
for f in range(F):
    nansf = []
    for n in range(N):
        for t in range(T):
            b = np.random.choice(100, 1)
            if b == 0 and np.isnan(X_all[n, f, t]) == 0:
                print(n, f, t)
                nansf.append([n, t])
                X_all[n, f, t] = np.nan
    nans.append(nansf)
X_old = X_all
X_linear = X_all

for f in range(F):
    batch = 12
    long, lat, x, y, z, xmin, ymin, xmax, ymax, gridx, gridy, gridz = setupKriging(coordsDf, batch)
    start = 0
    end = batch
    while (start < T):
        if (end > T):
            batch = T - start
            end = T
            long, lat, x, y, z, xmin, ymin, xmax, ymax, gridx, gridy, gridz = setupKriging(coordsDf, batch)
        v = X_all[:, f, start : end]
        xy_nidx, z_nidx = np.isnan(v).nonzero()
        data = np.column_stack((x, y, z, v.flatten()))
        data = np.delete(data, np.isnan(data).nonzero()[0], axis = 0)
        ok3d = OrdinaryKriging3D(data[:, 0], 
                                 data[:, 1], 
                                 data[:, 2], 
                                 data[:, 3], variogram_model = 'spherical')
        est, _ = ok3d.execute('grid', gridx, gridy, gridz)
        for xyid, zid in zip(xy_nidx, z_nidx):
            nanx = long[xyid]
            nany = lat[xyid]
            nanz = zid
            assert (np.isnan(X_all[xyid][f][start + zid]))
            X_all[xyid][f][start + zid] = est[zid][nany - ymin][nanx - xmin]
        start += batch
        end += batch
        print(f"Polno: {f}, start : {start}, end: {end}")

#savePMLevelsAllStationImputed(X_all,  N, F, T, coordsDf, "PMImputedNew")

print(x.shape, y.shape, z.shape)

val = X_all[:, 0, 0 : batch]
val = val.flatten()

data = np.column_stack((x, y, z, val))
data = np.delete(data, np.isnan(data).nonzero()[0], axis = 0)
ok3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model = 'spherical')

kval, _  = ok3d.execute('grid', gridx, gridy, gridz)
#imputeMissingVals(X_all, N, F, T, A)
#showConsecutiveNaNRanges(X_all, N, F, T)
#savePMLevelsAllStationImputed(X_all,  N, F, T, coordsDf, "PMImputedNew")
'''