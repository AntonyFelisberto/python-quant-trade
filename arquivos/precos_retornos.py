import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
from pylab import rcParams

yf.pdr_override()

rcParams["figure.figsize"] = 20,10

petr = web.get_data_yahoo("PETR4.SA",period="1y")["Adj Close"]
print(petr)
print(type(petr))

petr.plot()
plt.xlabel("Tempo - Data")
plt.ylabel("Preço")
plt.title("Petrobras")
plt.show()

ret = petr.pct_change()
ret.plot()
plt.xlabel("Tempo - Data")
plt.ylabel("Preço")
plt.title("Petrobras")
plt.show()