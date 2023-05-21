# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 00:55:40 2021

@author: Amartya Choudhury
"""
from utils import buildGraphFromJson, getPMLevelsImputed
import os
import pandas as pd
import shutil

def createParentDir(pDir):
    if (os.path.exists(pDir)):
        print(f"{pDir} exists. Entire directory tree will get deleted")
        shutil.rmtree(pDir)
    os.mkdir(pDir)
    return 

def saveSplit(path, train, test, polName, coordsDf):
    traindir = os.path.join(path, "train")
    testdir = os.path.join(path, "test")
    os.mkdir(traindir)
    os.mkdir(testdir)
    for idx, st in enumerate(coordsDf["Stations"]):
        filepath = os.path.join(traindir, st + ".csv")
        df = pd.DataFrame(train[idx].T, columns = [polName])
        df.to_csv(filepath)
        
        filepath = os.path.join(testdir, st + ".csv")
        df = pd.DataFrame(test[idx].T, columns = [polName])
        df.to_csv(filepath)
    return

N, A, coordsDf = buildGraphFromJson('coords.json')
X_all, F, T = getPMLevelsImputed('PMImputedNew', coordsDf, N)

trainPerCent = 80
ntrain = int((80 * T) / 100)
ntest = T - ntrain

x_train = X_all[:, :, 0 :ntrain]
x_test = X_all[:, :, ntrain : ]

pm10train = x_train[:, 0, :]
pm25train = x_train[:, 1, :]

pm10test = x_test[:, 0, :]
pm25test = x_test[:, 1, :]

pm10path = os.path.join(os.getcwd(), "PM10")
pm25path = os.path.join(os.getcwd(), "PM25")

createParentDir(pm10path)
createParentDir(pm25path)

saveSplit(pm10path,  pm10train, pm10test, "PM10", coordsDf)
saveSplit(pm25path,  pm25train, pm25test, "PM25", coordsDf)
