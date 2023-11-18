import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

import yfinance as yf
import pandas_datareader.data as web

def avalia_estacionaridade(x,cutoff = 0.01):
    pvalue = adfuller(x)[1]
    if pvalue < cutoff:
        print("Serie é estacionaria ", x.name)
        return True
    else:
        print("Serie nao é estacionaria ", x.name)
        return False
    

ruido = np.random.normal(0,1,100)
x = pd.Series(np.cumsum(ruido),name="X") + 50
x.plot(figsize=(20,10))
plt.show()

ruido_2 = np.random.normal(0,1,100)
y = x + 5 + ruido_2
pd.concat([x,y],axis=1).plot(figsize=(20,10))
x.plot(figsize=(20,10))
plt.show()

avalia_estacionaridade(x)
avalia_estacionaridade(y)

(y/x).plot(figsize=(20,10))
plt.axhline((y/x).mean(),color="red",linestyle="--")
plt.xlabel("tempo")
plt.legend(["ratio preço","média"])
plt.show()

z = y/x
z.name = "Z"
avalia_estacionaridade(z)