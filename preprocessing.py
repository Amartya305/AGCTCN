# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:22:22 2020

@author: Amartya Choudhury
"""
import pandas as pd
import numpy as np
from math import exp, log
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def linear_interpolate_missing(df, missing):
    for colname in df.columns:
        if (is)
        df[colname].replace(missing, np.nan, inplace = True)
        df[colname].interpolate('linear', inplace = True)
    return

def power_transform(df, old, new):
    df[new], lam = boxcox(df[old])
    print('Lambda value : {:.3f}'.format(lam))
    return lam

def diff_transform(df, old, new):
    df[new] = df[old] - df[old].shift(1)
    return

def season_transform(df, old, new, seasonal_lag = 48):
    df[new] = df[old] - df[old].shift(seasonal_lag)
    return
def inverse_season(train, train_col,test, test_col, new_col,lag = 48):
    test[new_col] = ""
    for r in range(len(train.index) - lag, len(train.index)):
        test.ix[r  - len(train.index) + lag, new_col] = train.iloc[r][train_col] + test.iloc[r  - len(train.index) + lag][test_col]
    for r in range(lag, len(test.index)):
        test.ix[r, new_col] = test.iloc[r - lag][new_col] + test.iloc[r][test_col]
    return

def inverse_trend(train, train_col,test, test_col, new_col):
    lag = 1
    test[new_col] = ""
    for r in range(len(train.index) - lag, len(train.index)):
        test.ix[r  - len(train.index) + lag, new_col] = train.iloc[r][train_col] + test.iloc[r  - len(train.index) + lag][test_col]
    for r in range(lag, len(test.index)):
        test.ix[r, new_col] = test.iloc[r - lag][new_col] + test.iloc[r][_col]
    return

def invert_boxcox(val, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)

def inverse_power(df, col, new_col, lam):
    for i in range(0, len(df.index)):
        df.ix[i,new_col] = invert_boxcox(df.iloc[i][col], lam)
    return

def standardize(train):
    transformer = StandardScaler()
    transformer.fit(train)
    transformed = transformer.transform(train)
    return transformer, transformed

def inverse_scaler(scaler, arr):
    return scaler.inverse_transform(arr)

def normalize(train):
    transformer = MinMaxScaler()
    transformer.fit(train)
    transformed = transformer.transform(train)
    return transformer, transformed


def get_pollutant_series(pr,pollutant, test_size = 300):
    df = pd.DataFrame(pr[pollutant], index = pr.index)
    df.columns = ['data']
    train = df.iloc[: -test_size, :]
    test = df.iloc[-test_size:,:]
    return train, test



def add_detrended_lagged_columns(df, window = 48, lag = 1):
    df['z_data'] = (df['data'] - df['data'].rolling(window = window).mean()) / df['data'].rolling(window = window).std()
    df['zp_data'] = ( df['z_data'] - df['z_data'].shift(lag))
    return

def dicky_fuller(df, col):
    result = adfuller(df[col].dropna(), autolag = 'AIC')
    print('Test statistic = {:.3f}'.format(result[0]))
    print('P value = {:.3f}'.format(result[1]))
    print('Critical Values')
    for k,v in result[4].items():
        print('\t{} : {}'.format(k, v))
    return


    
