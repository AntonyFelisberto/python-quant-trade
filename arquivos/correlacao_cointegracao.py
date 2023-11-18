import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller,coint
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


ret1 = np.random.normal(1,1,100)
ret2 = np.random.normal(2,1,100)

s1 = pd.Series(np.cumsum(ret1),name="S1")
s2 = pd.Series(np.cumsum(ret2),name="S2")

pd.concat([s1,s2],axis=1).plot(figsize=(15,7))
plt.show()

print(s1.corr(s2))
coint(s1,s2)

y2 = pd.Series(np.random.normal(0,1,800),name="Y2") + 20
y3 = y2.copy()
y3[0:100] = 30
y3[100:200] = 10
y3[200:300] = 30
y3[300:400] = 10
y3[400:500] = 30
y3[500:600] = 10
y3[600:700] = 30
y3[700:800] = 10

y2.plot(figsize=(20,10))
y3.plot()
plt.ylim([0,40])
plt.show()

print(y2.corr(y3))
print(coint(y2,y3))