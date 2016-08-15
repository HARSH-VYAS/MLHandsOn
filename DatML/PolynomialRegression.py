from pylab import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

x = np.array(pageSpeeds)
y = np.array(purchaseAmount)

xp = np.linspace(0, 7, 100)
print xp

#Polyfit function is there to fit any degree polynmial to fit the data.s
p4 = np.poly1d(np.polyfit(x, y, 4))

plt.scatter(x,y)
plt.plot(xp,p4(xp),c='r')
plt.show()

r2 = r2_score(y,p4(x))
print r2
