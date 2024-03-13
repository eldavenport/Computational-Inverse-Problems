# ensemble kalman filter using lorenz 96 

import numpy as np
import matplotlib.pyplot as plt
from L96Model import *
from Localization import * 

np.random.seed(1)

#set of L's and a's
alpha = np.array([0.05,0.075,0.1,0.125,0.15])
lengthscale = np.array([4,6,8,10])

# ----------------------------------------------------------------- Set Params ----------------------------------------------------------------------------------
dt = 0.05
T = 300
step = 0.2
ntimesteps = np.int32(T/step)

# ensemble size
Ne = 20
skip=2
# initial conditions, generate background sampels
nx = 40
ny = int(nx/skip)
gamma = 8

def getH(q,n):
    H = np.zeros((int(n/q),n))
    k = H.shape[0]
    for kk in range(0,k):
        jj = kk*q
        H[kk,jj] = 1
    return H

# ----------------------------------------------------------------- Set up ----------------------------------------------------------------------------------
# initialize ensemble as random sampels from the background
y_background = l96model(T=1000,dt=dt,nx=nx,gamma=gamma,x0=np.random.uniform(0,1,nx))
y_background = y_background[:,1000:] # remove some spin up time
nsamples = len(y_background[0,:])
ensemble_idx = np.int32(np.random.uniform(0,nsamples,Ne))
x_e0 = y_background[:,ensemble_idx] # generate an ensemble as random draw from some number of background samples
x0 = y_background[:,-1] # initial condition for true state

R = np.identity(ny)
l_r, U = np.linalg.eig(R)
Rsq = U @ np.diag(l_r**0.5)
H = getH(skip,nx) # to go from sample space to observation space

# ----------------------------------------------------------------- Filter ----------------------------------------------------------------------------------
RMSE = np.zeros((ntimesteps-1))
spread = np.zeros((ntimesteps-1))
x_e = np.zeros((nx,Ne)) 
Pa = np.zeros((nx,nx,ntimesteps))
mu_a = np.zeros((nx,ntimesteps))
x_e = x_e0
mu_a[:,0] = np.mean(x_e0,axis=1)
Pa[:,:,0] = np.eye(nx)

# generate the truth and data
x_true = l96model(T=T,dt=dt,nx=nx,gamma=gamma,x0=x0)
obs_idx = np.arange(0,int(T/dt),int(step/dt)) 
y = H @ x_true[:,obs_idx] + Rsq @ np.random.normal(0,1,(ny,ntimesteps))
x_true = x_true[:,obs_idx]

l = 4
a = 0.1
L = getL(l,nx)
for i in range(1,ntimesteps):
    #forecast wtih ensemble members
    for ensmem in range(0,Ne):
        x_e[:,ensmem] = l96model(T=0.2,dt=dt,nx=nx,gamma=gamma,x0=x_e[:,ensmem])[:,-1] # run forward for T=0.2, say true x0 is the last step of background

    # forecast mean
    mu_f = np.mean(x_e,axis=1)

    # inflation of ensemble members
    for ensemem in range(0,Ne):
        x_e[:,ensmem] = mu_f + np.sqrt(1 + a)*(x_e[:,ensmem]-mu_f)

    # forecast cov, with inflation and localization 
    Pf = np.cov(x_e)
    Pf = L * Pf

    # Kalman Gain
    arg = np.array(H @ Pf @ H.T + R)
    K = Pf @ H.T @ np.linalg.inv(arg)

    # analysis 
    for ensmem in range(0,Ne):
        perturbation = Rsq @ np.random.normal(0,1,ny)
        x_e[:,ensmem] = x_e[:,ensmem] +  K @ (y[:,i] - ((H @ x_e[:,ensmem]) + perturbation))

    # analysis covariance
    Pa[:,:,i] = (np.eye(nx) - K @ H) @ Pf
    # analysis mean
    mu_a[:,i] = np.mean(x_e,axis=1)

    #RMSE (true state and analysis mean) and spread
    spread[i-1] = np.sqrt((1/nx)*np.trace(Pa[:,:,i]))
    RMSE[i-1] = np.sqrt(np.mean(((mu_a[:,i]-x_true[:,i])**2))) # recompute this, compute mean error over ensemble memvers
    print(RMSE[i-1])
    print(spread[i-1])
# ----------------------------------------------------------------- Saving ----------------------------------------------------------------------------------
np.save('Homework6/RMSE.npy',RMSE)
np.save('Homework6/spread.npy',spread)
np.save('Homework6/x_true.npy',x_true)
np.save('Homework6/y.npy',y)
np.save('Homework6/mu_a.npy',mu_a)
