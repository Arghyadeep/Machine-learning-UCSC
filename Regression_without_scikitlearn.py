#dependencies
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

random.seed(9001)
xs = np.array([x for x in range(50)], dtype=np.float64)
ys = np.array(norm.rvs(10,5,50), dtype = np.float64)

def intercept_and_slope(xs,ys):
    m = ((mean(xs)*mean(ys))-mean(xs*ys))/((mean(xs)*mean(xs))-mean(xs*xs))

    b = mean(ys) - m*mean(xs)
    return m,b

m,b = intercept_and_slope(xs,ys)

regression_line = [(m*x)+b for x in xs]

plt.scatter(xs,ys,s=1)
plt.plot(xs,regression_line)
plt.show()
