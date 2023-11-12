import numpy as np
import matplotlib.pyplot as plt

xt = np.random.randn(10000)
plt.plot(xt)
plt.show()

xt.mean()
xt.var()
s = np.cumsum(xt)

plt.plot(s)
plt.show()