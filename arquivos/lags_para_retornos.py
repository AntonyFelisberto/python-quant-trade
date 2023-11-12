import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import train_test_split


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

dados = pd.DataFrame()

dados["close"] = petr
print(dados)

dados["retornos"] = ret
print(dados)

dados["lag1"] = dados["retornos"].pct_change(1)
print(dados)

dados["lag2"] = dados["retornos"].pct_change(2)
dados["lag3"] = dados["retornos"].pct_change(3)
dados["lag4"] = dados["retornos"].pct_change(4)
dados["lag5"] = dados["retornos"].pct_change(5)
print(dados)

dados = dados.dropna()
print(dados)

dados.iloc[:,2:].plot()
plt.show()

print(dados.shape)
print(dados.describe())
dados = dados[~dados.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
print(dados.shape)

y = dados["retornos"]
x = dados.iloc[:,2:]
print(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(x_train)

model = sm.OLS(y_train,x_train).fit()
predictions = model.predict(x_test)

print_model = model.summary()
print(print_model)

plot_acf(dados["close"])
plot_acf(dados["retornos"])