# -*- coding: utf-8 -*-
"""
Created on Sat May 15 17:21:31 2021

@author: Amartya Choudhury
"""

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import contextily as ctx
#from graph_utils import getEdgeList
from statsmodels.graphics.tsaplots import plot_acf
from pyproj import CRS
import seaborn as sns
from matplotlib import ticker
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from sklearn.linear_model import LinearRegression


def setTickFontSize(ax, size):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(size + 5)
        
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(size)
    
    
def reduceMajorTickSpacing(ax, xn, yn):
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        if (i % xn):
            tick.label.set_visible(False)
    for j, tick in enumerate(ax.yaxis.get_major_ticks()):
        if (j % yn):
            tick.label.set_visible(False)
   
def plotPredictions(yTrue, yPred, coordsDf, N):
    fig, axes = plt.subplots(N, figsize = (25, 500))
    for i in range(N):
        axes[i].set_title(coordsDf["Stations"][i], fontsize = 30)
        axes[i].set_xlabel('Time', fontsize = 20)
        axes[i].set_ylabel('PM10 level', fontsize = 20)
        axes[i].plot(yTrue[i][0], label = 'Original', color = 'black')
        axes[i].plot(yPred[i][0], label = 'Forecast', color = 'red')
        axes[i].legend(fontsize = 20)
        for tick in axes[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
            tick.set_visible(True)
        for tick in axes[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in axes[i].get_xticklabels():
            tick.set_visible(True)
    plt.tight_layout()
    fig.autofmt_xdate()

def plotLocations(edges, pts, coordsStMap):
    fig,ax = plt.subplots(figsize = (30,30))
    edgeDf = gpd.GeoDataFrame(geometry = edges)
    ax = edgeDf.plot(ax = ax, linewidth = 1.2, linestyle ='-', color = 'red', zorder=1, label = 'Links', alpha = 1)
    ptsDf = None
    colors = ['mediumspringgreen', 'cyan', 'salmon',  'plum', 'khaki']
    for i in range(len(pts)):
        ptsDf = gpd.GeoDataFrame(geometry = pts[i])
        ptsDf.crs = CRS("EPSG:4326")
        bbox = dict(boxstyle ="round", facecolor = colors[i])
        ax = ptsDf.plot(ax = ax, markersize = 130, color = colors[i], zorder=2)
        for x, y in zip(ptsDf.geometry.x, ptsDf.geometry.y):
            ax.annotate("ST" + str(coordsStMap.get((x, y))), xy = (x, y),  xytext=(5, 5), fontsize = 20, textcoords="offset pixels", bbox = bbox)
    ctx.add_basemap(ax, crs=ptsDf.crs.to_string(), source=ctx.providers.Stamen.Toner)
    ax.set_xlabel("Longitude", fontsize = 30)
    ax.set_ylabel("Latitude", fontsize = 30)
    #ax.set_title("Pollution monitoring stations, Delhi", fontsize = 40)
    setTickFontSize(ax, 20)
    ax.legend(fontsize = 30)
    ax.spines['bottom'].set_color('1')
    ax.spines['top'].set_color('1')
    ax.spines['right'].set_color('1')
    ax.spines['left'].set_color('1')
    plt.show()

def plotAcf(pm10Avg, pm25Avg):
    fig, axes = plt.subplots(1, 2, figsize = (35, 10))
    
    plot_acf(pm10Avg, axes[0], lags = np.arange(0, 500, 5), color = 'black',
             vlines_kwargs = {'color':'black'})
    
    axes[0].set_xlabel('Quarter hourly time lag', fontsize = 30)
    axes[0].set_ylabel('Autocorrelation', fontsize = 30)
    axes[0].set_title('Autocorrelation plot of spatially averaged PM10 levels',
                      fontsize = 35)
    setTickFontSize(axes[0], 20)
    
    plot_acf(pm25Avg, axes[1], lags = np.arange(0, 500, 5), color = 'black',
             vlines_kwargs = {'color':'black'})
    
    axes[1].set_xlabel('Quarter hourly time lag',fontsize = 30)
    axes[1].set_ylabel('AutoCorrelation',fontsize = 30)
    axes[1].set_title('Autocorrelation plot of spatially averaged PM2.5 levels',
                      fontsize = 35)
    setTickFontSize(axes[1], 20)
    
    plt.tight_layout()
    fig.autofmt_xdate()

def plotCovariogram(C, N, breaks, poln):
    plt.figure(figsize=(20,20))
    sns.set(font_scale=2)
    ax = sns.heatmap(C, cmap = "YlGnBu", square = True, norm = plt.Normalize())
    idxnames = ["ST" + str(i) for i in range(1, N + 1)]
    ax.invert_yaxis()
    for b in breaks:
        if (b == (N - 1)):
            break
        ax.vlines(b +1,-0.5, N, color = 'black')
        ax.hlines(b + 1, -0.5, N, color = 'black')
    ax.set_xlabel("Stations", fontsize = 30)
    ax.set_ylabel("Sations" , fontsize = 30)        
    #ax.set_title(f"Lag-0 spatio-temporal correlation matrix for {poln}" , fontsize = 35)
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(idxnames, rotation = 45)
    ax.set_yticks(np.arange(N))
    ax.set_yticklabels(idxnames, rotation = 45)
    setTickFontSize(ax, 20)
    reduceMajorTickSpacing(ax, 5, 5)
    
def plotSpatialDistr(X1, X2, X3, X4, nm1 = "baseline", 
                     nm2 = "3 hours later",
                     nm3 = "6 hours later",
                     nm4 = "12 hours later"):
    fig, axes = plt.subplots(1, 2, figsize = (35, 10))
    idx = np.arange(len(X1))
    idxnames = ["ST" + str(i) for i in range(1, len(X1) + 1)]
    axes[0].plot(idx, X1[:, 0], 'o', color = 'green')
    axes[0].plot(idx, X1[:, 0], color = 'green', label = nm1)
    axes[0].plot(idx, X2[:, 0], 'o', color = 'deepskyblue')
    axes[0].plot(idx, X2[:, 0], color = 'deepskyblue', label = nm2)
    axes[0].plot(idx, X3[:, 0], 'o', color = 'purple')
    axes[0].plot(idx, X3[:, 0], color = 'purple', label = nm3)
    axes[0].plot(idx, X4[:, 0], 'o', color = 'red')
    axes[0].plot(idx, X4[:, 0], color = 'red', label = nm4)
    axes[0].legend(loc = 'upper left')
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(idxnames, rotation = 45)
    axes[0].set_xlabel("Station ID", fontsize = 30)
    axes[0].set_ylabel("Hourly mean PM10 levels  (ug / m^3)" , fontsize = 30)
    axes[0].set_title("Evolution of spatial distribution of PM10 levels with time", fontsize = 35)    
    setTickFontSize(axes[0], 20)
    axes[1].plot(idx, X1[:, 1], 'o', color = 'green')
    axes[1].plot(idx, X1[:, 1], color = 'green', label = nm1)
    axes[1].plot(idx, X2[:, 1], 'o', color = 'deepskyblue')
    axes[1].plot(idx, X2[:, 1], color = 'deepskyblue', label = nm2)
    axes[1].plot(idx, X3[:, 1], 'o', color = 'purple')
    axes[1].plot(idx, X3[:, 1], color = 'purple', label = nm3)
    axes[1].plot(idx, X4[:, 1], 'o', color = 'red')
    axes[1].plot(idx, X4[:, 1], color = 'red', label = nm4)
    axes[1].legend(loc = 'upper left')
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(idxnames, rotation = 45)
    axes[1].set_xlabel("Station ID", fontsize = 30)
    axes[1].set_ylabel("Hourly mean PM2.5 levels (ug / m^3)" , fontsize = 30)
    axes[1].set_title("Evolution of spatial distribution of PM2.5 levels with time", fontsize = 35)
    setTickFontSize(axes[1], 20)
    
def plotTemporal(X, X_q):
    fig, ax = plt.subplots(1, 1, figsize = (35, 10))
    ax.plot(X, label = "Mean")
    ax.plot(X_q[0], label = "10th quantile")
    ax.plot(X_q[1], label = "90th quantile")
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 35233, 2840)))
    months = ["Sep'19", "Oct'19", "Nov'19", "Dec'19", "Jan'20", "Feb'20", "Mar'20", "Apr'20", "May'20", 
              "Jun'20", "Jul'20", "Aug'20", "Sep'20"]
    ax.set_xticklabels(months)
    ax.set_xlabel("Month of the Year", fontsize = 30)
    ax.set_ylabel(u"PM 2.5 level (\u03bcg / m^3)", fontsize = 30)
    #ax.set_title("Distribution of PM2.5 concentrations over the year", fontsize = 35)
    setTickFontSize(ax, 20)
    ax.legend()

def plotStationwiseCORR(corr, N):
    fig, ax = plt.subplots(1, 1, figsize = (35, 10))
    stnames = ["ST" + str(i) for i in range(1, N +1)]
    ax.plot(corr["LSTM"], label = "LSTM")
    ax.plot(corr["ConvLSTM"], label = "ConvLSTM")
    ax.plot(corr["GCLSTM"], label = "GCLSTM")
    ax.plot(corr["AGCTCN"], label = "AGCTCN")
    ax.legend(loc = 'upper right')
    ax.set_xticks(idx)
    ax.set_xticklabels(idxnames, rotation = 45)
    axes[0].set_xlabel("Station ID", fontsize = 30)
    
def plotOneBox(ax, metricData, metricName, polName):
      ax = sns.boxplot(data = pd.DataFrame(metricData), ax =ax, palette="Spectral")
      ax = sns.swarmplot(data = pd.DataFrame(metricData), ax =ax, color=".25")
      ax.set_ylabel(polName + " " + metricName, fontsize = 25)
      setTickFontSize(ax, 20)
      
def boxplotMetrices(RMSE, NRMSE, CORR, MAE, R2):
    fig, axes = plt.subplots(2, 5, figsize = (35, 15))
    Pols = ["$PM_{10}$", "$PM_{2.5}$"]
    metrics = ["RMSE", "NRMSE", "CORR", "MAE", "$R^2$"]
    metricData = [RMSE, NRMSE, CORR, MAE, R2]
    for i in range(len(Pols)):
      for j in range(len(metrics)):
          plotOneBox(axes[i, j], metricData[j][i], metrics[j], Pols[i])
    plt.tight_layout()
    fig.autofmt_xdate()       

def densityScatter( x , y,  xlim, ylim, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = False )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    ax.scatter( x, y, c=z, s = 0.5, **kwargs, cmap = 'hot' )
   # ax.spines['left'].set_position('zero')
    #ax.spines['bottom'].set_position('zero')
    ax.set_xlim(left = 20, right = xlim)
    ax.set_ylim(bottom = 20, top = ylim)
    norm = Normalize(vmin = np.min(data), vmax = np.max(data))
    cbar = plt.colorbar(cm.ScalarMappable(norm = norm, cmap = 'hot'), ax=ax)
    cbar.ax.set_ylabel('Density', fontsize = 20)
    cbar.ax.tick_params(labelsize = 17)
    return ax

def abline(ax, slope, intercept, c, linestyle, label):
    """Plot a line from slope and intercept"""
    #axes = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, linestyle = linestyle, c = c, label = label)
    return ax
    
def correlationPlotOne(fig, ax, pol, model, X, Y, i):
    X = X.flatten()
    Y = Y.flatten()
    ax = densityScatter(X, Y, 400 - 200*i, 400 -200*i, ax = ax)
    regr = LinearRegression().fit(X.reshape(-1, 1), Y.reshape(-1, 1))
    ax = abline(ax, regr.coef_[0][0], regr.intercept_[0], 'blue', '--', 
                "$\hat{Z} = $" + 
            f"{round(regr.intercept_[0], 2)} + " + 
            f"{round(regr.coef_[0][0], 2)}" +
            "$Z$")
    ax = abline(ax, 1, 0, 'black', '-', '$\hat{Z} = Z$')
    ax.set_xlabel("Observed " + pol + " ($\mu g/m^3$)", fontsize = 25)
    ax.set_ylabel("Estimated " + pol + " ($\mu g/m^3$)", fontsize = 25)
    ax.set_title(model, fontsize = 25)
    ax.legend(fontsize = 20, loc = 'upper left')
    setTickFontSize(ax, 20)
      
def correlationPlot(X, Y):
    Pols = ["$PM_{10}$", "$PM_{2.5}$"]
    #plt.subplots_adjust(hspace = 0.8)
    fig, ax = plt.subplots(2, 4, figsize = (35, 15), constrained_layout=True)
    for i, pol in enumerate(Pols):
        for j, model in enumerate(X[i].keys()):
            correlationPlotOne(fig, ax[i, j], pol, model, X[i][model], Y[i][model], i)
    #plt.tight_layout()
    fig.autofmt_xdate()

def barplotOne(ax, i, Pols, metrics, metricname):
    barWidth = 0.25
    br = 2*np.arange(len(Pols))
    #colors = ["#003f5c", "#444e86", "#955196", "#dd5182", "#ff6e54", "#ffa600"]
    colors = ["lightcoral", "maroon", "olive", "darkgreen", "dodgerblue", "darkviolet"]
    for j, model in enumerate(metrics[0].keys()):
        if j != 0:
            br = [x + barWidth for x in br]
        ax.bar(br, [metrics[0][model], metrics[1][model]], width = barWidth / 2, edgecolor = "black", label = model, color = colors[j])
    #ax.set_xlabel("Pollutant", fontsize = 25)
    ax.set_ylabel(metricname, fontsize = 30)
    ax.set_xticks([2*r + 3*barWidth for r in range(len(Pols))])
    ax.set_xticklabels(Pols)
    if i == 0:
        l = ax.legend(fontsize = 20, loc = 'upper left')
    elif i == 1:
        l = ax.legend(fontsize = 20, loc = 'lower left')
    else:
        l = ax.legend(fontsize = 20, loc = 'upper left')
    for lh in l.legendHandles: 
        lh.set_alpha(1)
    setTickFontSize(ax, 25)
    
def barplotAblation(RMSE, CORR, MAE):
    metrics = [RMSE, CORR, MAE]
    metricNms = ["RMSE", "CORR", "MAE"]
    Pols = ["$PM_{10}$", "$PM_{2.5}$"]
    fig, ax = plt.subplots(1, 3, figsize = (40, 10), constrained_layout=True)
    for i in range(len(metricNms)):
        barplotOne(ax[i], i, Pols, metrics[i], metricNms[i])
    #fig.autofmt_xdate()