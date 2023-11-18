import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller

rcParams["figure.figsize"]=20,10

def gerar_pontos_normal(params):
    mu = params[0]
    sigma = params[1]
    ponto = np.random.normal(mu,sigma)
    return ponto

params = (0,1)
T = 100
A = pd.Series(index=range(T))
A.name = "A"
for t in range(T):
    A[t] = gerar_pontos_normal(params)

plt.plot(A)
plt.xlabel("tempo - t")
plt.ylabel("valores")
plt.legend(["serie temporal A"])
plt.show()

T = 100
B = pd.Series(index=range(T))
B.name = "B"
for t in range(T):
    params = (t*0.1,1)
    B[t] = gerar_pontos_normal(params)

plt.plot(B)
plt.xlabel("tempo - t")
plt.ylabel("valores")
plt.legend(["serie temporal B"])
plt.show()

m = np.mean(B)

plt.plot(B)
plt.hlines(m,0,len(B),linestyles="dashed",colors="r")
plt.xlabel("Time")
plt.ylabel("value")
plt.legend(["Series B","Mean"])
plt.show()

def avalia_estacionaridade(x,cutoff = 0.01):
    pvalue = adfuller(x)[1]
    if pvalue < cutoff:
        print("Serie é estacionaria ", x.name)
        return True
    else:
        print("Serie nao é estacionaria ", x.name)
        return False
    
avalia_estacionaridade(A)