# Homework 1 - building the state estimation algorithm

import numpy as np
import matplotlib.pyplot as plt
from LSQCost import cost_fxn_resid
from scipy.optimize import least_squares
from L96Model import l96model
from os import getcwd

np.random.seed(105)
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

y_morzfeld = np.array([7.0529, 3.3435, -1.8879, 0.5335, 0.0905, -4.4914, 1.4622, 11.1491, -0.7031, 5.5337, -2.7527, 5.7205, 0.6421, 4.9910, -1.6307, -0.0879, 7.4980, 0.6805, 6.7904, 2.0327])
R = np.identity(ny)

# Make background cov matrix B using a long simulation (several thousand time units)
b = l96model(T=5000,dt=dt,nx=nx,gamma=gamma,x0=x0)
B = np.cov(b)
mu = b.mean(axis=1) # take the time mean 

# compute square root of covariance matrices
l, U = np.linalg.eig(R)
R_sq = U @ np.diag(l**-0.5)
l, U = np.linalg.eig(B)
B_sq = U @ np.diag(l**-0.5)

## BEGIN HOMEWORK1
def nonlinsq_l96(y, guess):

    M = lambda x0: l96model(T=0.2,dt=0.01,nx=40,x0=x0,gamma=8)
    resid = lambda x0: cost_fxn_resid(M,R_sq,B_sq,mu,y,x0)
    sol = least_squares(fun=resid, x0=guess, method='lm')
    return sol.x

RMSE_xestx0 = np.zeros(50)
RMSE_yestytrue_full = np.zeros(50)
RMSE_yestydata = np.zeros(50)
for it in range(50):
    print(it)
    guess = np.random.normal(0,1,nx)
    best_est = nonlinsq_l96(y,guess)
    y_best_est = l96model(T=0.2,dt=dt,nx=nx,gamma=gamma,x0=best_est)
    y_est = y_best_est[idx,-1]

    xaxis = range(nx)
    xaxis_data = range(0,nx,2)

    RMSE_xestx0[it] = np.linalg.norm(x0-best_est,ord=2)/np.sqrt(len(x0))
    RMSE_yestytrue_full[it] = np.linalg.norm(y2[:,-1]-y_best_est[:,-1],ord=2)/np.sqrt(len(y2[:,-1]))
    RMSE_yestydata[it] = np.linalg.norm(y-y_est,ord=2)/np.sqrt(len(y))

fig, ax = plt.subplots(nrows=3,constrained_layout=True)
ax[0].hist(RMSE_xestx0,10)
ax[0].set_xlabel('RMS Error')
ax[0].set_ylabel('Total Instances')
ax[0].set_title('Initital Condition Error (X0, X*)')

ax[1].hist(RMSE_yestytrue_full,10)
ax[1].set_xlabel('RMS Error')
ax[1].set_ylabel('Total Instances')
ax[1].set_title('T=0.2 Error (Xt, X*t)')

ax[2].hist(RMSE_yestydata,10)
ax[2].set_xlabel('RMS Error')
ax[2].set_ylabel('Total Instances')
ax[2].set_title('Y Error (Y, Yhat)')
plt.savefig('RMSE.png')
plt.show()

# fig, ax = plt.subplots()
# ax.plot(xaxis,x0,'b')
# ax.plot(xaxis,best_est,'r')
# ax.set_xlabel('Index')
# ax.set_ylabel('Value of Xi')
# ax.legend(['True X0','Estimated X0'])
# ax.set_title('X0 estimated from Nonlinear LS')
# plt.savefig(pwd + '/Homework1/Xest_X0.png')
# plt.close()

# fig, ax = plt.subplots(figsize=(10,10))
# ax.plot(xaxis,y2[:,-1],'b')
# ax.plot(xaxis,y_best_est[:,-1],'r')
# ax.errorbar(xaxis_data,y,yerr=2*np.ones(len(y)),fmt='go')
# ax.set_xlabel('Index')
# ax.set_ylabel('Value of Xi')
# ax.legend(['True Y','Y from estimated X0','Data'])
# ax.set_title('Y at T=0.2 from X0 NLS Estimate')
# plt.savefig(pwd + '/Homework1/Yest_Ytrue.png')

RMSE_ymorzydata = np.linalg.norm(y_morzfeld-y_est,ord=2)/np.sqrt(len(y_morzfeld))

print('RMSE0 = ' + str(RMSE_xestx0))
print('RMSET = ' + str(RMSE_yestytrue_full))
print('RMSEy = ' + str(RMSE_yestydata))

# RMSE0 = 2.0774681588614468
# RMSET = 1.6058249383641647
# RMSEy = 0.32576440817575975

#Morzfeld data 
# RMSEy = 0.3244332909124902