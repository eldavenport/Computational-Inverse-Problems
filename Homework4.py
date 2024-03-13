#implement RTO to estimate posterior distribution 

import numpy as np
import matplotlib.pyplot as plt
from LSQCost import cost_fxn_resid
from scipy.optimize import least_squares
from L96Model import l96model
from numba import njit
import time 
# np.random.seed(2)

import multiprocessing
multiprocessing.set_start_method("fork")
from multiprocessing import Pool
from multiprocessing import cpu_count
import os

os.environ["OMP_NUM_THREADS"] = "1"
ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

# for each sample solve the optimization problem with perturbed y and mu
R_sq = np.load('Code/Homework2/HW2_R_sq.npy')
B_sq = np.load('Code/Homework2/HW2_B_sq.npy')
Rsq = np.load('Code/Homework2/HW2_Rsq.npy')
Bsq = np.load('Code/Homework2/HW2_Bsq.npy')
y_data = np.load('Code/Homework2/HW2_y.npy')
x0 = np.load('Code/Homework2/HW2_x0.npy')
y_true = np.load('Code/Homework2/HW2_ytrue.npy')
mu = np.load('Code/Homework2/HW2_mu.npy')

nx = len(x0)
ny = len(y_data)
dt = 0.01 
gamma = 8
idx = np.arange(0,nx,2)
nSamples = 10000

# @njit
def nonlinsq_l96(y, guess):

    M = lambda x0: l96model(T=0.2,dt=0.01,nx=40,x0=x0,gamma=8)
    resid = lambda x0: cost_fxn_resid(M,R_sq,B_sq,mu,y,x0)
    sol = least_squares(fun=resid, x0=guess, method='lm',ftol=10**-6)
    return sol.x

# randomize then optimize, generate a new random data and a new random mu and then 
# find the best estimate
samples = np.zeros((nx,nSamples))
samples_t2 = np.zeros((nx,nSamples))
samples_y = np.zeros((ny,nSamples))
startTime = time.time()

with Pool() as pool:
    for it in range(nSamples):
        y_p = y_data + Rsq @ np.random.normal(0,1,len(y_data)) # generating statistics w Rsq cov
        mu_p = mu + Bsq @ np.random.normal(0,1,len(mu))
        best_est = nonlinsq_l96(y_p,mu_p)
        samples[:,it] = best_est
        y_best_est = l96model(T=0.2,dt=dt,nx=nx,gamma=gamma,x0=best_est)
        samples_t2[:,it] = y_best_est[:,-1]
        samples_y[:,it] = y_best_est[idx,-1]
        if it == 100:
            time_now = time.time()
            total_time = time_now - startTime
            print("time to 100 samples: {0:.1f} seconds".format(total_time))

time_now = time.time()
total_time = time_now - startTime
print("time to finish samples: {0:.1f} seconds".format(total_time))

np.save('Code/Homework4/samples_x.npy',samples)
np.save('Code/Homework4/samples_t2.npy',samples_t2)
np.save('Code/Homework4/samples_y.npy',samples_y)

print('done! and saved!')

# xaxis = range(nx)
# xaxis_data = range(0,nx,2)
# RMSE_xestx0[it] = np.linalg.norm(x0-best_est,ord=2)/np.sqrt(len(x0))
# RMSE_yestytrue_full[it] = np.linalg.norm(y2[:,-1]-y_best_est[:,-1],ord=2)/np.sqrt(len(y2[:,-1]))
# RMSE_yestydata[it] = np.linalg.norm(y-y_est,ord=2)/np.sqrt(len(y))