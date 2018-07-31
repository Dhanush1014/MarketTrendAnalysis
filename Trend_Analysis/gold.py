#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:34:06 2018

@author: Dhanush
"""

from sklearn.linear_model import LinearRegression 

# pandas and numpy are used for data manipulation 

import pandas as pd 

import numpy as np 

# matplotlib and seaborn are used for plotting graphs 

import matplotlib.pyplot as plt 

import seaborn 

# fix_yahoo_finance is used to fetch data 

import fix_yahoo_finance as yf
# Read data 

Df = yf.download('GLD','2008-01-01','2017-12-31')
Df=Df[['Close']]
Df= Df.dropna() 
Df.Close.plot(figsize=(10,5)) 
plt.ylabel("Gold ETF Prices")
plt.show()
Df['S_3'] = Df['Close'].shift(1).rolling(window=3).mean() 

Df['S_9']= Df['Close'].shift(1).rolling(window=9).mean() 
Df= Df.dropna() 

X = Df[['S_3','S_9']] 

X.head()


y = Df['Close']
y.head()
t=.8 

t = int(t*len(Df)) 

# Train dataset 

X_train = X[:t] 

y_train = y[:t]  

# Test dataset 

X_test = X[t:] 

y_test = y[t:]

linear = LinearRegression().fit(X_train,y_train) 

print ("Gold ETF Price =", round(linear.coef_[0],2),

"* 3 Days Moving Average", round(linear.coef_[1],2), 

"* 9 Days Moving Average +", round(linear.intercept_,2))
predicted_price = linear.predict(X_test)  

predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])  

predicted_price.plot(figsize=(10,5))  

y_test.plot()  

plt.legend(['predicted_price','actual_price'])  

plt.ylabel("Gold ETF Price")  

plt.show()
r2_score = linear.score(X[t:],y[t:])*100  

float("{0:.2f}".format(r2_score))