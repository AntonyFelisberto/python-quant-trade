import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller,coint
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

def avalia_estacionaridade(x,cutoff = 0.01):
    pvalue = adfuller(x)[1]
    if pvalue < cutoff:
        print("Serie é estacionaria ", x.name)
        return True
    else:
        print("Serie nao é estacionaria ", x.name)
        return False

T = 100

x1 = np.random.normal(0,1,T)
x1 = np.cumsum(x1)
x1 = pd.Series(x1)
x1.name = "x1"

x2 = x1 + np.random.normal(0,1,T)
x2.name = "x2"

plt.plot(x1)
plt.plot(x2)
plt.xlabel("time")
plt.ylabel("series values")
plt.legend([x1.name, x2.name])
plt.show()

avalia_estacionaridade(x1)
avalia_estacionaridade(x2)

z = x2 - x1
z.name = "z"

plt.plot(z)
plt.xlabel("time")
plt.ylabel("valor da serie")
plt.legend(["Z"])
plt.show()
avalia_estacionaridade(z)

score, pvalue, _ = coint(x1,x2)
print(score,pvalue,sep=" ")
