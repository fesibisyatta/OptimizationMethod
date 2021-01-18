#%%
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
#%%
# 観測度数
obs = np.array([
    [4, 2, 3],
    [8, 4, 6],
    [6, 3, 6],
])
#%%
# 各行, 列での平均
y_i = np.mean(obs, axis = 1)*3 # 行方向の平均
y_j = np.mean(obs, axis = 0)*3 # 列方向の平均
n = 42
#%%
# 最尤推定量
p_tilder = np.array([np.array([k*y_i[l] for k in y_j]) for l in range(3)])/n**2
p_tilder
#%%
# 理論度数
y_ij_hat = n * p_tilder
y_ij_hat
#%%
# χ二乗適合度統計量
chi_sq = ((obs - y_ij_hat)**2/y_ij_hat).sum()
chi_sq
# %%
## 多項分布の一様性検定
obs = np.array([
    [148, 444, 86],
    [111, 352, 49],
    [645, 1911, 328],
    [165, 771, 119],
    [383, 1829, 311],
    [96, 293, 47],
    [98, 330, 58],
    [199, 874, 155],
    [59, 199, 30],
    [262, 1320, 236]
])
#%%
def theo_freq(obs):
    # 分割表の各軸における合計値
    row_sums = np.sum(obs, axis = 1)
    col_sums = np.sum(obs, axis = 0)
    # 全サンプルの数
    n = obs.sum()
    # 各項目における最尤推定量
    p_tilder = np.array([np.array([k * row_sums[l] for k in col_sums]) for l in range(obs.shape[0])])/n**2
    # 各項目における理論度数
    y_hat_ij = p_tilder*n
    return y_hat_ij

def chi_sq(obs, freq):
    return {'degree of freedom': (obs.shape[0] - 1)*(obs.shape[1] - 1),'chi square goodness of fit':((obs-freq)**2/freq).sum()}

# %%
theo_freq(obs)
# %%
chi_sq(obs, theo_freq(obs))
# %%

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x = np.array([8.2, 7.5, 8.7, 8.4, 9.6])
dataset = np.array([4.95, 5.02, 5.08, 4.90, 5.12,
                    5.19, 4.80, 4.87, 4.98, 4.58, 
                    4.81, 5.18, 5.25, 5.05, 4.79, 
                    5.57, 5.01, 5.57, 5.01, 4.81, 
                    5.13, 5.04, 5.13, 5.16, 4.69])

def t_dist(x , theta):
    return 1/(1 + (x - theta)**2)

def likelihood(theta, datalist, k):
    w = 0
    for i in range(k):
        w = t_dist(datalist, theta)
        theta = (w*datalist).sum()/w.sum()
    return w, theta

def trim(x, k):
    x = sorted(x)[k:-k]
    return np.mean(x)

def huber(x):
    x = np.array(sorted(x))
    n = len(x)
    s = (np.floor(n/2) + 1)/2
    if isinstance(s, float):
        x_s = (x[int(s + 0.5) -1] + x[int(s - 0.5) -1])*0.5
        x_ns1 = (x[int(n - s + 1 + 0.5) -1] + x[int(n - s + 1 - 0.5) -1])*0.5
        d = x_ns1 - x_s
    else:
        d = x[-s] - x[s]
    c = 0.5 * d
    x_med = np.median(x)
    cond_u = np.array(x <= x_med + c)
    cond_l = np.array(x >= x_med - c)
    both = cond_u*cond_l
    m = (both).sum()
    S = x[both].sum()
    m_U = (~cond_u).sum()
    m_L = (~cond_l).sum()
    theta_H = (S + c * (m_U - m_L))/m
    return theta_H


for i in range(20):
    plt.scatter(i, likelihood(theta = 8.4, datalist = x, k = i)[1])
plt.show()

llist = [likelihood(theta = 8.4, datalist = x, k = i)[1] for i in range(200)]
plt.plot(list(range(len(llist))), llist)
plt.show()

trim(dataset, 3)

huber(dataset)
