# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:16:57 2023

@author: CHINNU BABY
"""
#importing libraries
import pandas as pd
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import scipy.optimize as opt
from sklearn.cluster import KMeans
from scipy.stats import norm
import seaborn as sns
from scipy.optimize import curve_fit
import itertools as iter
#inserting the indicator id's
indicator1 = ["EN.ATM.CO2E.PC","EG.USE.ELEC.KH.PC"]
indicator2 = ["EN.ATM.METH.KT.CE","EG.ELC.ACCS.ZS"]
#selecting differnt country codes
country_code = ['AUS','BGD','CAN','DNK','ETH']
#function to read the data which is taken from the world bank
def read(indicator,country_code):
    df = wb.data.DataFrame(indicator, country_code, mrv=30)
    return df
#calling the csv file
file = "co2 emission.csv"
#a function to read indicator1 and country_code
data = read(indicator1, country_code)
#removing YR and giving new index names to data
data.columns = [i.replace('YR','') for i in data.columns]
data=data.stack().unstack(level=1)
data.index.names = ['Country', 'Year']
data.columns
#creating another dataframe
data1 = read(indicator2, country_code)
#removing YR and giving index names to data1
data1.columns = [i.replace('YR','') for i in data1.columns]
data1=data1.stack().unstack(level=1)
data1.index.names = ['Country', 'Year']
data1.columns
#creating indices for dt1 and dt2
dt1 = data.reset_index()
dt2 = data1.reset_index()
dt = pd.merge(dt1, dt2)
dt
dt.drop(['EG.USE.ELEC.KH.PC'], axis = 1, inplace = True)
dt.drop(['EG.ELC.ACCS.ZS'], axis = 1, inplace = True)
dt
dt["Year"] = pd.to_numeric(dt["Year"])
def norm_df(df):
    y = df.iloc[:,2:]
    df.iloc[:,2:] = (y-y.min())/ (y.max() - y.min())
    return df
dt_norm = norm_df(dt)
dt_norm
df_fit = dt_norm.drop('Country', axis = 1)
k = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(df_fit)
sns.scatterplot(data=dt_norm, x="Country", y="EN.ATM.CO2E.PC", hue=k.labels_)
plt.legend()
plt.show()
def err_ranges(x, func, param, sigma):
#initiate arrays for lower and upper limits
  lower = func(x, *param)
  upper = lower
  uplow = []
#list to hold upper and lower limits for parameters
  for p,s in zip(param, sigma):
    pmin = p - s
    pmax = p + s
    uplow.append((pmin, pmax))
    pmix = list(iter.product(*uplow))
  for p in pmix:
    y = func(x, *p)
    lower = np.minimum(lower, y)
    upper = np.maximum(upper, y)
  return lower, upper
dt1 = dt[(dt['Country'] == 'AUS')]
dt1
val = dt1.values
x, y = val[:, 1], val[:, 2]
def fct(x, a, b, c):
 return a*x**2+b*x+c
prmet, cov = opt.curve_fit(fct, x, y)
dt1["pop_log"] = fct(x, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
plt.plot(x, dt1["pop_log"], label="Fit")
plt.plot(x, y, label="Data")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in Australia")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x, fct, prmet, sigma)
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fct, prmet, sigma)
print("2030 between", low, "and", up)
dt2 = dt[(dt['Country'] == 'CAN')]
dt2
val2 = dt2.values
x2, y2 = val2[:, 1], val2[:, 2]
def fct(x, a, b, c):
 return a*x**2+b*x+c
prmet, cov = opt.curve_fit(fct, x2, y2)
dt2["pop_log"] = fct(x2, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
plt.plot(x2, dt2["pop_log"], label="Fit")
plt.plot(x2, y2, label="Data")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in Canada")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x2, fct, prmet, sigma)
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fct, prmet, sigma)
print("2030 between", low, "and", up)
dt3 = dt[(dt['Country'] == 'ETH')]
dt3
val3 = dt3.values
x3, y3 = val3[:, 1], val3[:, 2]
def fct(x, a, b, c):
 return a*x**2+b*x+c
prmet, cov = opt.curve_fit(fct, x3, y3)
dt3["pop_log"] = fct(x3, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
plt.plot(x3, dt3["pop_log"], label="Fit")
plt.plot(x3, y3, label="Data")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in Ethiopia")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x3, fct, prmet, sigma)
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fct, prmet, sigma)
print("2030 between", low, "and", up)
