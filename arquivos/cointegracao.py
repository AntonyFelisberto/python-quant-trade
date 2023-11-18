import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller,coint
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

import yfinance as yf
import pandas_datareader.data as web
yf.pdr_override()

def avalia_estacionaridade(x,cutoff = 0.01):
    pvalue = adfuller(x)[1]
    if pvalue < cutoff:
        print("Serie é estacionaria ", x.name)
        return True
    else:
        print("Serie nao é estacionaria ", x.name)
        return False

symbols = ["^BVSP","ABEV3.SA","ITSA4.SA","PETR4.SA","VALE3.SA"]

portifolio = web.get_data_yahoo(symbols,period="1y")["Close"]

portifolio = portifolio.rename(columns={"ABEV3.SA":"ABEV3","ITSA4.SA":"ITSA4","PETR4.SA":"PETR4","VALE3.SA":"VALE3"})

x1 = portifolio["PETR4"]
x2 = portifolio["ITSA4"]

plt.plot(x1)
plt.plot(x2)
plt.xlabel("tempo")
plt.ylabel("valores")
plt.legend([x1.name,x2.name])
plt.show()

x1 = sm.add_constant(x1)
results = sm.OLS(x2,x1).fit()

x1 = x1["PETR4"]
print(results.params)
print(results.summary())

b = results.params["PETR4"]
z = x2 - b * x1
z.name = "z"

plt.plot(z.index,z.index)
plt.xlabel("time")
plt.ylabel("series value")
plt.legend([z.name])
plt.show()

avalia_estacionaridade(z)

print(coint(x1,x2))