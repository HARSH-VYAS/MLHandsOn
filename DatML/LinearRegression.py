import numpy as np
from pylab import *
from scipy import stats
import matplotlib.pyplot as plt

def predict(x):
    return slope * x + intercept

pageSpeeds = np.random.normal(4.0, 1.0, 100)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.1, 100)) * 3

slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)
# R squared value shows a real good fit.
print r_value ** 2

fitLine = predict(pageSpeeds)

plt.scatter(pageSpeeds, purchaseAmount,c='r')
plt.plot(pageSpeeds, fitLine, c='b')
plt.show()
