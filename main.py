# -*- coding: utf-8 -*-
"""
Created on Wed May  5 03:19:54 2021

@author: Amartya Choudhury
"""

from utils import *
from models import *
import cProfile
import pstats
from visuals import *


N, A, coordsDf = buildGraphFromJson('coords.json')
X_all, F, T = getPMLevelsAllStations(coordsDf, N)

#plotLocations(N, A, coordsDf)
#pm10Acf, pm25Acf = buildAndPlotAcf(X_all)

imputeMissingVals(X_all, N, F, T, A)
showConsecutiveNaNRanges(X_all, N, F, T)
savePMLevelsAllStationImputed(X_all,  N, F, T, coordsDf)
C = getCovariogram(X_all[:, 0, :])

nTrain = 34233
X_train, X_test = trainTestSplit(X_all, nTrain, T)

#extract weekly, hourly and recent components
Tp = 12
Th = 24
Td = 12
Tw = 24
q = 24 * 4
Xh, Xd, Xw, Y = extractComponents(X_train, Tp, Th, Td, Tw, q)

Xh = getSingleFeature(Xh, 0)
Xd = getSingleFeature(Xd, 0)
Xw = getSingleFeature(Xw, 0)
Y = getSingleFeature(Y, 0)

device = setupCUDA()

model = trainASTGCN(Xh, Xd, Xw, Y, N, A, Th, Td , Tw, Tp, device, name = 'ASTGCN_PM10_400_')

Yp, Yt = test(model, X_all, Th, Td, Tp, q, nTrain, device, path = 'ASTGCN_complete_07_02_07_14-05-21_PM10_400.pt')     
plotPredictions(Y_t, Y_pred, coordsDf, N)


def trainOneBatch():
    for e in range(1):
        loss = 0.0
        valLoss = 0.0
        for Xh, Xd, Xw, Y in trainLoader:
            Yp = model(Xh, Xd, Xw, A)
            lossT = mse(Yp, Y)
            optimizer.zero_grad()
            lossT.backward()
            optimizer.step()
            loss += lossT.item()
            print(lossT.item())
            break


profile = cProfile.Profile()
profile.runcall(trainOneBatch)
ps = pstats.Stats(profile)
ps.print_stats()
'''



