import numpy as np
import scipy
import pylab as pl
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

from scipy import constants as C
print(C.c)
print(C.h)
print(C.physical_constants['electron mass'])
print(C.mile)
print(C.inch)
print(C.gram)
print(C.pound)

import scipy.special as S
print(S.log1p(1e-20))

m = np.linspace(0.1, 0.9, 4)
n = np.linspace(-10, 10, 100)
results = S.ellipj(n[:, None], m[None, :])
print([y.shape for y in results])

#spatial空间算法库
##取相邻最近点
from scipy import spatial
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import pylab as pl
import numpy as np

x = np.sort(np.random.rand(100))
idx = np.searchsorted(x, 0.5)
print(idx)
print(x[idx], x[idx - 1])

np.random.seed(42)
N = 100
points = np.random.uniform(-1, 1, (N, 2))
kd = spatial.cKDTree(points)
targets = np.array([(0, 0), (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)])
dist, idx = kd.query(targets, 3)

#拟合与优化optimize
##非线性方程组求解
import pylab as pl
import numpy as np
from scipy import optimize
from math import sin, cos
import matplotlib as mpl
mpl. rcParams['font.sans-serif'] = ['SimHei']

def f(x):
    x0, x1, x2 = x.tolist()
    return [
        5*x1+3,
        4*x0*x0 - 2*sin(x1*x2),
        x1*x2 - 1.5
    ]

result = optimize.fsolve(f, [1, 1, 1])
print(result)
print(f(result))

def j(x):
    x0, x1, x2 = x.tolist()
    return [[0, 5, 0],
    [8 * x0, -2 * cos(x1 * x2), -2 * x1 * cos(x1 *x2)],
    [0, x2, x1]]

result = optimize.fsolve(f, [1,1,1], fprime = j, full_output = True)#fprime用于计算雅克比矩阵
print(result)

##最小二乘拟合
import numpy as np
from scipy import optimize

X = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
Y = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])

def residuals(p):
    k, b = p
    return Y - (X*k + b)

r = optimize.leastsq(residuals, [1, 0])
print(r)
k, b = r[0]
print(k, b)

