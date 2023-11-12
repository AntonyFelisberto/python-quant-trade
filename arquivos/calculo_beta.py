import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas_datareader.data as web

yf.pdr_override()

symbols = ["^BVSP","ABEV3.SA","ITSA4.SA","PETR4.SA","VALE3.SA"]

portifolio = web.get_data_yahoo(symbols,period="1y")["Close"]

print(portifolio)
portifolio = portifolio.rename(columns={"ABEV3.SA":"ABEV3","ITSA4.SA":"ITSA4","PETR4.SA":"PETR4","VALE3.SA":"VALE3"})
portifolio.plot(figsize=(20,10))
plt.show()

portifolio.iloc[:,0:4].plot(figsize=(20,10))
plt.show()

retornos = portifolio.pct_change()
print(retornos)

retornos = portifolio.pct_change().dropna()
print(retornos)

retornos.plot(figsize=(20,10))
plt.show()

retornos.corr()
print(retornos.corr())

sns.heatmap(retornos.corr())
sns.heatmap(retornos.corr(),annot=True)

betas = retornos.cov()/retornos["^BVSP"].var()
print(betas)