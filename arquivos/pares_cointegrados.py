import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller,coint
yf.pdr_override()

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n,n))
    pvalue_matrix = np.ones((n,n))
    keys = data.keys()
    pairs = [] 
    for i in range(n):
        for j in range(i+1,n):
            s1 = data[keys[i]]
            s2 = data[keys[j]]
            result = coint(s1, s2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i,j] = score
            pvalue_matrix[i,j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i],keys[j]))
    return score_matrix,pvalue_matrix,pairs

def plot_pairs(d2,par):
    (d2[par[0]]/np.mean(d2[par[0]])).plot()

    (d2[par[1]]/np.mean(d2[par[1]])).plot()
    plt.legend(par)
    plt.show()

def zscore(series):
    return (series - series.mean()) / np.std(series)

def desenha_ratio(d2,par):
    data = d2
    s1 = data[par[0]]
    s2 = data[par[0]]

    score,pvalue, _ = coint(s1,s2)
    print(f"p valor = {pvalue}")
    ratios = s1/s2

    zscore(ratios).plot(figsize=(15,7))

    plt.axhline(zscore(ratios).mean(),color="black")
    plt.axhline(1.0,color="red",linestyle="--")
    plt.axhline(-1.0,color="green",linestyle="--")
    plt.legend(["ratio z score","mean","+1","-1"])
    plt.show()


symbols = ["^BVSP", "ABEV3.SA", "ITSA4.SA", "PETR4.SA","PETR3.SA", "VALE3.SA","B3SA3.SA","BBAS2.SA","BBSC3.SA","BBDC4.SA"]
portfolio = web.get_data_yahoo(symbols, start="2019-03-01", end="2020-08-17")["Close"]

scores,pvalues,pairs = find_cointegrated_pairs(portfolio.fillna(0))
#scores,pvalues,pairs = find_cointegrated_pairs(portfolio.dropna())

par = pairs[0]
s1 = portfolio[par[0]]
s2 = portfolio[par[1]]

score,pvalue,_ = coint(s1,s2)
print(f"p valor {pvalue}")
ratios = s1/s2

ratios.plot()
plt.axhline(ratios.mean())

plot_pairs(portfolio,pairs[0])
plot_pairs(portfolio,pairs[1])

data = portfolio.fillna(0).copy()
ratios = data.loc[:,"ABEV3.SA"]/data.loc[:,"BBSC3.SA"]
print(len(ratios))
train = ratios[:150]
test = ratios[150:]

ratios_mavg5 = train.rolling(window=5,center=False).mean()

ratios_mavg60 = train.rolling(window=60,center=False).mean()

std_60 = train.rolling(window=60,center=False).std()

zscore_60_5 = (ratios_mavg5 - ratios_mavg60) / std_60

plt.figure(figsize=(15,7))
plt.plot(train.index,train.values)
plt.plot(ratios_mavg5.index,ratios_mavg5.values)
plt.plot(ratios_mavg60.index,ratios_mavg60.values)

plt.legend(["Ratio","5D ratio MA","60D ratio Ma"])

plt.ylabel("ratio")
plt.show()

std_60 = train.rolling(window=60,center=False).std()
std_60.name="std 60d"

zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
zscore_60_5.name = "z-score"

plt.figure(figsize=(15,7))
zscore_60_5.plot()
plt.axhline(0,color="black")
plt.axhline(1.0,color="red",linestyle="--")
plt.axhline(-1.0,color="green",linestyle="--")
plt.legend(["Rolling ratio z-score","mean","+1","-1"])
plt.show()

plt.figure(figsize=(20,10))

train[60:].plot()

buy = train.copy()
sell = train.copy()

buy[zscore_60_5 > -1] = 0
buy[zscore_60_5 < 1] = 0

buy[60:].plot(color="g",linestyle="None",marker="^")
sell[60:].plot(color="r",linestyle="None",marker="v")

x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,ratios.min(),ratios.max()))
plt.legend(["Ratios","Buy Signal","Sell Signal"])
plt.show()

plt.figure(figsize=(20,10))

par = pairs[0]
qnt = 700
s1 = data[par[0]].iloc[:qnt]
s1 = data[par[1]].iloc[:qnt]

s1[60:].plot(color="b")
s1[60:].plot(color="b")
buy_r = 0 * s1.copy()
sell_r = 0 * s1.copy()

buy_r[buy!=0] = s1[buy!=0]
sell_r[buy!=0] = s2[buy!=0]

buy_r[sell!=0] = s2[sell!=0]
sell_r[sell!=0] = s1[sell!=0]

buy_r[60:].plot(color="g",linestyle="None",marker="^")
sell_r[60:].plot(color="g",linestyle="None",marker="v")

plt.legend([par[0],par[1],"buy sinal","sell sinal"])
plt.show()