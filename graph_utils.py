# -*- coding: utf-8 -*-
"""
Created on Wed May  5 03:22:46 2021

@author: Amartya Choudhury
"""
import numpy as np
import haversine as hs
from math import exp
import torch

def computeDistance(coords, i, j, sigma = 10, epsilon = 0.5):
    if (i == j):
        return 0
    pi = coords[i]
    pj = coords[j]
    dij = hs.haversine(pi, pj)
    Wij = exp(- (dij * dij) / (sigma * sigma))
    Wij = Wij if Wij > epsilon else 0
    return Wij
    
def buildAdjMatrixFromCoords(coords, thres = 0.5):
    N = len(coords)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            A[i][j] = computeDistance(coords, i, j, epsilon = thres)
    return N, A

def buildGraphFromCoords(df, colnames, thres = 0.5):
    coords = []
    for index, row in df.iterrows():
        coords.append((row[colnames[0]], row[colnames[1]]))
    N, A = buildAdjMatrixFromCoords(coords, thres)
    return N, A

def getGCNParams(A):
    batch, row, col = (A > 0).nonzero().t()
    index = torch.stack([row, col], dim = 0)
    weight = A[batch, row, col]
    return batch, index, weight 

def getEdgeIndexAndWeight(A):
    edgeIndex = (A > 0).nonzero().t()
    row, col = edgeIndex
    edgeWeight = A[row, col]
    return edgeIndex, edgeWeight

