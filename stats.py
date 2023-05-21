# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 15:19:37 2021

@author: Amartya Choudhury
"""
from utils import buildGraphFromJson, getPMLevelsImputed, getCovariogram, buildAndPlotAcf
from shapely.geometry import Point, LineString
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from visuals import plotLocations, plotCovariogram, plotSpatialDistr, plotTemporal
from datetime import datetime

def getStationClusters(coordsDf, nclusters = 5):
    coords = np.column_stack((np.array(coordsDf["Latitude"]), np.array(coordsDf["Longitude"])))
    k = KMeans(nclusters, random_state = 42).fit(coords)
    cluster = np.column_stack((np.arange(0, N), k.labels_))
    clusterDf = pd.DataFrame(cluster)
    clusterDf.sort_values(by = 1, inplace = True, ignore_index = True)
    #perm = np.array(clusterDf[0])
    return clusterDf

def getMapPoints(coordsDf, clusterDf, N, nclusters = 5):
    ptclstrArr = []
    coordsStMap = dict()
    clstrEnds = []
    for clstr in range(nclusters):
        ptclstr = []
        clstrEnd = 0
        for idx in range(N):
            if (clusterDf.loc[idx, 1] == clstr):
                stidx = clusterDf.loc[idx, 0]
                x= coordsDf.loc[stidx, "Longitude"]
                y = coordsDf.loc[stidx, "Latitude"]
                pt = Point(x, y)
                coordsStMap[(x, y)] = idx + 1
                clstrEnd = idx
                ptclstr.append(pt)
        clstrEnds.append(clstrEnd)
        ptclstrArr.append(ptclstr)
    return ptclstrArr, clstrEnds, coordsStMap
    
def getMapEdges(coordsDf, A):
    frm, to = (A > 0).nonzero()
    edges = []
    for i in range(frm.shape[0]):
        a = frm[i]
        b = to[i]
        pa = Point(coordsDf.loc[a, "Longitude"], coordsDf.loc[a, "Latitude"])
        pb = Point(coordsDf.loc[b, "Longitude"], coordsDf.loc[b, "Latitude"])
        e = LineString([pa, pb])
        edges.append(e)
    return edges

def negFilter(x):
    if (x < 0):
        return 0
    return x

def rstrip(x):
    x = x.rstrip(", Delhi - DPCC")
    x = x.rstrip(", Delhi - CPCB")
    x = x.rstrip(", Delhi - IM")
    return x

N, A, coordsDf = buildGraphFromJson('coords.json')
X_all, F, T = getPMLevelsImputed('PMImputedNew', coordsDf, N)

stCluster = getStationClusters(coordsDf)
pts, breaks, coordsStMap = getMapPoints(coordsDf, stCluster, N)
edges = getMapEdges(coordsDf, A)
plotLocations(edges, pts, coordsStMap)

perm = np.array(stCluster[0])
X_clus = X_all[perm]

C = getCovariogram(X_clus[:, 0, :], lag = 0)
plotCovariogram(C, N, breaks, "PM10")
C = getCovariogram(X_clus[:, 1, :], lag = 0)
plotCovariogram(C, N, breaks, "PM2.5")

buildAndPlotAcf(X_all)

X_base = X_clus[:, :, 0 : 4]
X_3hrs = X_clus[:, :, 3*4 : 4*4]
X_6hrs = X_clus[:, :, 6*4 : 7*4]
X_12hrs = X_clus[:, :, 12*4 : 13*4]

X_base_means = np.mean(X_base, axis = -1)
X_3hrs_means = np.mean(X_3hrs, axis = -1)
X_6hrs_means = np.mean(X_6hrs, axis = -1)
X_12hrs_means = np.mean(X_12hrs, axis = -1)
plotSpatialDistr(X_base_means, X_3hrs_means, X_6hrs_means, X_12hrs_means)

#plotSpatialCorrMap(X_base_means, X_3hrs_means, X_6hrs_means, X_12hrs_means)

summaryDf  = coordsDf.copy()
summaryDf = summaryDf.reindex(perm)
summaryDf["mean"] = 0.0
summaryDf["std"] = 0.0
summaryDf["min"] = 0.0
summaryDf["25%"] = 0.0
summaryDf["50%"] = 0.0
summaryDf["75%"] = 0.0
summaryDf["max"] = 0.0
summaryDf["skewness"] = 0.0
summaryDf["kurtosis"] = 0.0
#coordsDfP

for n in range(N):
    stats = pd.DataFrame(X_clus[n, 0, :]).describe().transpose()
    summaryDf.at[n, "mean"] = stats["mean"]
    summaryDf.at[n, "std"] = stats["std"]
    summaryDf.at[n, "min"] = stats["min"]
    summaryDf.at[n, "25%"] = stats["25%"]
    summaryDf.at[n, "50%"] = stats["50%"]
    summaryDf.at[n, "75%"] = stats["75%"]
    summaryDf.at[n, "max"] = stats["max"]
    summaryDf.at[n, "skewness"] = pd.DataFrame(X_clus[n, 0, :]).skew()
    summaryDf.at[n, "kurtosis"] = pd.DataFrame(X_clus[n, 0, :]).kurtosis()
    
summaryDf["ID"] = np.arange(1, 28)
summaryDf["min"] = summaryDf["min"].apply(negFilter)
summaryDf["Stations"] = summaryDf["Stations"].apply(rstrip)
summaryDf = summaryDf.round(2)
summaryDf = summaryDf[summaryDf.columns[0:-1].insert(0, 'ID')]

summaryDf.to_csv('PM10summary.csv', index = False)


summaryDf  = coordsDf.copy()
summaryDf["Perm"] = perm
summaryDf .sort_values(by = "Perm", inplace = True, ignore_index = True)
summaryDf .drop("Perm", inplace = True, axis = 1)
summaryDf["mean"] = 0.0
summaryDf["std"] = 0.0
summaryDf["min"] = 0.0
summaryDf["25%"] = 0.0
summaryDf["50%"] = 0.0
summaryDf["75%"] = 0.0
summaryDf["max"] = 0.00
summaryDf["skewness"] = 0.0
summaryDf["kurtosis"] = 0.0
#coordsDfP

for n in range(N):
    stats = pd.DataFrame(X_clus[n, 1, :]).describe().transpose()
    summaryDf.at[n, "mean"] = stats["mean"]
    summaryDf.at[n, "std"] = stats["std"]
    summaryDf.at[n, "min"] = stats["min"]
    summaryDf.at[n, "25%"] = stats["25%"]
    summaryDf.at[n, "50%"] = stats["50%"]
    summaryDf.at[n, "75%"] = stats["75%"]
    summaryDf.at[n, "max"] = stats["max"]
    summaryDf.at[n, "skewness"] = pd.DataFrame(X_clus[n, 1, :]).skew()
    summaryDf.at[n, "kurtosis"] = pd.DataFrame(X_clus[n, 1, :]).kurtosis()
    
summaryDf["ID"] = np.arange(1, 28)
summaryDf["min"] = summaryDf["min"].apply(negFilter)
summaryDf["Stations"] = summaryDf["Stations"].apply(rstrip)
summaryDf = summaryDf.round(2)
summaryDf = summaryDf[summaryDf.columns[0:-1].insert(0, 'ID')]

summaryDf.to_csv('PM25summary.csv', index = False)


meanPM25 = np.mean(X_clus[:, 1, :], axis = 0)
quantPM25 = np.quantile(X_clus[:, 1, :], q = [0.1, 0.9], axis = 0)
dates = np.array(pd.read_csv("Delhi/Ashok Vihar, Delhi - DPCC.csv")["From Date"])
plotTemporal(meanPM25, quantPM25)