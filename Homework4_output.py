# homework 3 - nonlinear UQ

import numpy as np
import matplotlib.pyplot as plt
from L96Model import l96model

# multiprocessing.set_start_method("fork")
# from multiprocessing import Pool
# from multiprocessing import cpu_count
# import os

# os.environ["OMP_NUM_THREADS"] = "1"
# ncpu = cpu_count()
# print("{0} CPUs".format(ncpu))

np.random.seed(5)

Rsq = np.load('Code/Homework2/HW2_Rsq.npy')
Bsq = np.load('Code/Homework2/HW2_Bsq.npy') # square root matrix, not inverse square root
y = np.load('Code/Homework2/HW2_y.npy')
x0 = np.load('Code/Homework2/HW2_x0.npy')
y_true = np.load('Code/Homework2/HW2_ytrue.npy')
samples_x = np.load('Code/Homework4/samples_x.npy')
samples_t2 = np.load('Code/Homework4/samples_t2.npy')
samples_y = np.load('Code/Homework4/samples_y.npy')
x_star = np.load('Code/Homework2/HW2_x.npy')
linear_best_est = np.load('Code/Homework2/HW2_linearBestEst.npy')

print(samples_x.shape)

nx = len(x0)
ny = nx//2
nsamples = samples_x.shape[1]

plt.hist(samples_x[0,:], 100, histtype="step")
plt.hist(samples_x[5,:], 100, histtype="step")
plt.hist(samples_x[10,:], 100, histtype="step")
plt.hist(samples_x[15,:], 100, histtype="step")
plt.hist(samples_x[20,:], 100, histtype="step")
plt.hist(samples_x[25,:], 100, histtype="step")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([])
plt.savefig('Code/Homework4/Hist.png',format='png')
plt.show()
plt.close()

mu_post = samples_x.mean(axis=1)
                       
fig, ax = plt.subplots()
# ax.plot(range(nx), mu, color='#8f99fb',linewidth=0.3,label='_nolegend_')
ax.plot(range(nx), mu_post,'m-*',linewidth=1.5)
ax.plot(range(nx), x0,'b',linewidth=1.5)
ax.plot(range(nx), x_star,'g',linewidth=1.5)
ax.legend(['X_RTO','X0','X*'])
ax.set_title('RTO Posterior Mean, compared to truth and optimization')
ax.set_xlabel('X Index')
ax.set_ylabel('Magnitude')
plt.savefig('Code/Homework4/Posterior_mean.png')
plt.show()
plt.close()

gamma = 8
dt = 0.01
y_best_est_RTO = l96model(T=0.2,dt=dt,nx=nx,gamma=gamma,x0=mu_post)
nsubsamples = 10000
sample_idxs = np.int64(np.floor(np.random.uniform(0,nsamples,nsubsamples)))
subsamples = samples_t2[:,sample_idxs]

axis_data = range(0,nx,2)

fig, ax = plt.subplots()
ax.plot(range(nx),subsamples,color='#8f99fb',linewidth=0.3,label='_nolegend_')
ax.plot(range(nx),y_best_est_RTO[:,-1],'g',linewidth=1.5)
ax.plot(range(nx),linear_best_est,'r',linewidth=1.5)
ax.plot(range(nx),y_true,'b',linewidth=1.5)
ax.errorbar(axis_data,y,yerr=2*np.ones(len(y)),fmt='o',color='k')
ax.legend(['XT RTO','XT Gauss-Newton','XT0','Ytrue'])
ax.set_title('Uncertainty Quanitification at time T=0.2')
ax.set_xlabel('X Index')
ax.set_ylabel('Values')
plt.savefig('Code/Homework4/RTO_UQ.png')
plt.show()

subsamples = samples_x[:,sample_idxs]

fig, ax = plt.subplots()
ax.plot(range(nx),subsamples,color='#8f99fb',linewidth=0.3,label='_nolegend_')
ax.plot(range(nx),mu_post,'g',linewidth=1.5)
ax.plot(range(nx),x_star,'r',linewidth=1.5)
ax.plot(range(nx),x0,'b',linewidth=1.5)
ax.legend(['X RTO','X Gauss-Newton','X0'])
ax.set_title('Uncertainty Quanitification for X0')
ax.set_xlabel('X Index')
ax.set_ylabel('Values')
plt.savefig('Code/Homework4/RTO_UQ_X0.png')
plt.show()
plt.close()

# RMSE's 
RMSE_xestx0 = np.zeros(nsamples)
RMSE_yestytrue_full = np.zeros(nsamples)
RMSE_yestydata = np.zeros(nsamples)
for i in range(0,nsamples):
    # RMSE samples x versus x0
    RMSE_xestx0[i] = np.linalg.norm(x0-samples_x[:,i],ord=2)/np.sqrt(len(x0))
    # RMSE samples t2 versus y_true
    RMSE_yestytrue_full[i] = np.linalg.norm(y_true-samples_t2[:,i],ord=2)/np.sqrt(len(y_true))
    # RMSE samples y versus y_data
    RMSE_yestydata[i] = np.linalg.norm(y-samples_y[:,i],ord=2)/np.sqrt(len(y))

fig, ax = plt.subplots(figsize=(5,8),nrows=3,constrained_layout=True)
ax[0].hist(RMSE_xestx0,100)
ax[0].set_xlabel('RMS Error')
ax[0].set_ylabel('Total Instances')
ax[0].set_title('Initital Condition Error (X0, X*)')

ax[1].hist(RMSE_yestytrue_full,100)
ax[1].set_xlabel('RMS Error')
ax[1].set_ylabel('Total Instances')
ax[1].set_title('T=0.2 Error (Xt, X*t)')

ax[2].hist(RMSE_yestydata,100)
ax[2].set_xlabel('RMS Error')
ax[2].set_ylabel('Total Instances')
ax[2].set_title('Y Error (Y, Yhat)')
plt.savefig('Code/Homework4/RMSE.png')
plt.show()