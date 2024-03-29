# Homework 1 - building the state estimation algorithm

import numpy as np
import matplotlib.pyplot as plt
from LSQCost import cost_fxn
from scipy.optimize import least_squares
from L96Model import l96model
from os import getcwd

np.random.seed(2)
pwd = getcwd()

## FROM HOMEWORK 0 
nx = 40
ny = 20
dt = 0.01
gamma = 8
x_noise = np.random.uniform(0,1,nx)

y1 = l96model(T=1000,dt=dt,nx=nx,gamma=gamma,x0=x_noise)
x0 = y1[:,-1]

# create synthetic data using the result above (y) as the initial condition
y2 = l96model(T=0.2,dt=dt,nx=nx,gamma=gamma,x0=x0)

# select every other state variable from the last time step
idx = np.arange(0,nx,2)
y_true = y2[idx,-1]
y = y_true + np.random.normal(0,1,len(y_true))
R = np.identity(ny)

# Make background cov matrix B using a long simulation (several thousand time units)
b = l96model(T=5000,dt=dt,nx=nx,gamma=gamma,x0=x0)
B = np.cov(b)
mu = b.mean(axis=1) # take the time mean 

# compute square root of covariance matrices
L = np.linalg.cholesky(R)
R_sq = np.linalg.inv(L)
L = np.linalg.cholesky(B)
B_sq = np.linalg.inv(L)

fig, ax = plt.subplots()
pos1 = ax.imshow(B_sq,cmap='PuOr',vmin=-1*np.max(abs(B_sq)),vmax=np.max(abs(B_sq)))
im_ratio = B_sq.shape[0]/B_sq.shape[1]
cbar2 = fig.colorbar(pos1,ax=ax,fraction=0.047*im_ratio)
ax.set_xlabel('X index')
ax.set_ylabel('Y index')
ax.set_title('inv(B^1/2)')
plt.savefig(pwd + '/Homework1/Binv_cholesky.png')
plt.close()

guess = np.random.normal(0,1,nx)

## BEGIN HOMEWORK1
def nonlinsq_l96(y, guess):

    M = lambda x0: l96model(T=0.2,dt=0.01,nx=40,x0=x0,gamma=8)
    resid = lambda x0: cost_fxn(M,R_sq,B_sq,mu,y,x0)
    sol = least_squares(fun=resid, x0=guess)
    return sol.x

best_est = nonlinsq_l96(y,guess)
y_best_est = l96model(T=0.2,dt=dt,nx=nx,gamma=gamma,x0=best_est)
y_est = y_best_est[idx,-1]

xaxis = range(nx)
xaxis_data = range(0,nx,2)

fig, ax = plt.subplots()
ax.plot(xaxis,x0,'b')
ax.plot(xaxis,best_est,'r')
ax.set_xlabel('Index')
ax.set_ylabel('Value of Xi')
ax.legend(['True X0','Estimated X0'])
ax.set_title('X0 estimated from Nonlinear LS')
plt.savefig(pwd + '/Homework1/Xest_X0_cholesky.png')
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(xaxis,y2[:,-1],'b')
ax.plot(xaxis,y_best_est[:,-1],'r')
ax.errorbar(xaxis_data,y,yerr=2*np.ones(len(y)),fmt='go')
ax.set_xlabel('Index')
ax.set_ylabel('Value of Xi')
ax.legend(['True Y','Y from estimated X0','Data'])
ax.set_title('Y at T=0.2 from X0 NLS Estimate')
plt.savefig(pwd + '/Homework1/Yest_Ytrue_cholesky.png')

RMSE_xestx0 = np.linalg.norm(x0-best_est,ord=2)/np.sqrt(len(x0))
RMSE_yestytrue_full = np.linalg.norm(y2[:,-1]-y_best_est[:,-1],ord=2)/np.sqrt(len(y2[:,-1]))
RMSE_yestydata = np.linalg.norm(y-y_est,ord=2)/np.sqrt(len(y))

print('RMSE0 = ' + str(RMSE_xestx0))
print('RMSET = ' + str(RMSE_yestytrue_full))
print('RMSEy = ' + str(RMSE_yestydata))

# RMSE0 = 1.9068947584692797
# RMSET = 1.589390742149101
# RMSEy = 0.22779668815393145
