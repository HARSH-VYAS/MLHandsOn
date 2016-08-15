#Regression problem to find out the future values of Stock prices. Forecasting/Prediction problem

import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

#choosing style to plot the graph
style.use('ggplot')

# Getting stock prices from Quandl.com for google inc.
df = quandl.get("WIKI/GOOGL")   

#Selecting some column data to consider for our Regression problem
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]] 

# Finding out some Feature values to consider
df['HL_pct'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100.0 

df['Change_pct'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.0

# Defining the feature matrix
df = df[['Adj. Close','HL_pct','Change_pct','Adj. Volume',]]

forecast_col = 'Adj. Close'         #Deciding which coloum to use as Our label Adj.Close is a good fit.
df.fillna(-99999,inplace=True)      #Replacing Null values with some random value suppose -99999 

# We can predict # days future price using this.
forecast_out = int (math.ceil(0.01*len(df)))
print("Days",forecast_out)

# Shifting last 10 values of Adj. Close/ Forecast coloum up and set it under ForecastLabel coloum 
df['ForecastLabel'] = df[forecast_col].shift(-forecast_out)

# Define X(Features),y(Labels)
X =  np.array(df.drop('ForecastLabel',1))

#scaling the new data.
X = preprocessing.scale(X)

# The values of X for which we will predict the output
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
df.dropna(inplace=True)

df.dropna(inplace=True)
y = np.array(df['ForecastLabel'])


#Train, Test data creation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#fit the train data
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)
#Serialization of any python object is called pickeling
with open('pickellR','wb') as f:
    pickle.dump(clf,f)

pickle_in = open('pickellR','rb')
clf= pickle.load(pickle_in)





accuracy = clf.score(X_test,y_test)
#print (accuracy)

#Prediction function
forecast_set = clf.predict(X_lately)

print(forecast_set,accuracy, forecast_out)

#plotting the graph date vs forecast value

df['Forecast']= np.nan
last_date = df.iloc[-1].name #last date
last_unix = last_date.timestamp()#timestamp on last date
one_day=86400 #No of seconds
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()





