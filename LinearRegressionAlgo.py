from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6,7],dtype=np.float64) 
#ys= np.array([5,4,5,7,5,6,5],dtype=np.float64)

def createDataset(hmp, variance, step=2,correlation = False):
    val= 1
    ys= []
    for i in range(hmp):
        y = val + random.randrange (-variance, variance)
        ys.append(y)
        if correlation and correlation ==  'pos':
            val +=step
        elif correlation and correlation ==  'neg':
            val-=step
    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64), np.array(ys,dtype=np.float64)


#Best fit line calculation.
def bestFitSlopeAndIntercept(xs,ys):
    m=( ( mean(xs) * mean(ys)- mean(xs*ys) ) / ( mean(xs)**2 - mean(xs**2) ) )
    b = mean(ys)- m * mean(xs)    
    return m, b
#Squared error Calculation

def squaredError(ys_org,ys_ln):
    return sum((ys_ln-ys_org)**2)

# Define the coefficient of the determination

def coeffOfDetermination(ys_org,ys_ln):
    y_meanln = [mean(ys_org) for y in ys_org]
    sqarred_err_regr = squaredError(ys_org,ys_ln)
    sqarred_err_mean = squaredError(ys_org,y_meanln)
    return 1 - (sqarred_err_regr/sqarred_err_mean)

xs, ys = createDataset(40,10,2, correlation = 'pos')


m,b = bestFitSlopeAndIntercept(xs,ys)

#print(m,b)

predictX = 8
predictY =  (m*predictX)+b

# Regression line...
regression_line = [ (m*x) + b  for x in xs]

# Coefficient of Determination

r_squared = coeffOfDetermination(ys, regression_line)

print(r_squared)
plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.scatter(predictX,predictY, color='g')
plt.show()
