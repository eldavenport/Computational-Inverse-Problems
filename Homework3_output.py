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

x_star = np.load('Code/Homework2/HW2_x.npy')
J = np.load('Code/Homework2/HW2_J.npy')
R_sq = np.load('Code/Homework2/HW2_R_sq.npy')
B_sq = np.load('Code/Homework2/HW2_B_sq.npy')
y = np.load('Code/Homework2/HW2_y.npy')
x0 = np.load('Code/Homework2/HW2_x0.npy')
y_true = np.load('Code/Homework2/HW2_ytrue.npy')
linear_best_est = np.load('Code/Homework2/HW2_linearBestEst.npy')
samples = np.load('Code/Homework3/samples.npy')
print(samples.shape)

nx = len(x_star)
ny = nx//2
nsamples = samples.shape[0]

plt.hist(samples[:, 0], 100, histtype="step")
plt.hist(samples[:, 5], 100, histtype="step")
plt.hist(samples[:, 10], 100, histtype="step")
plt.hist(samples[:, 15], 100, histtype="step")
plt.hist(samples[:, 20], 100, histtype="step")
plt.hist(samples[:, 25], 100, histtype="step")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([])
plt.savefig('Code/Homework3/Hist.png',format='png')
plt.show()
plt.close()

mu_post = samples.mean(axis=0)
                       
fig, ax = plt.subplots()
# ax.plot(range(nx), mu, color='#8f99fb',linewidth=0.3,label='_nolegend_')
ax.plot(range(nx), mu_post,'m-*',linewidth=1.5)
ax.plot(range(nx), x0,'b',linewidth=1.5)
ax.plot(range(nx), x_star,'g',linewidth=1.5)
ax.legend(['X_MCMC','X0','X*'])
ax.set_title('MC Hammer Posterior Mean, compared to truth and optimization')
ax.set_xlabel('X Index')
ax.set_ylabel('Magnitude')
plt.savefig('Code/Homework3/Posterior_mean.png')
plt.show()
plt.close()

gamma = 8
dt = 0.01
nsubsamples = 1000
idx = np.arange(0,nx,2)

# select nsamples from samples (select 200 random rows)
sample_idxs = np.int64(np.floor(np.random.uniform(0,nsamples,nsubsamples)))
subsamples = samples[sample_idxs,:].T

print('running samples forward')
XT_sample = np.zeros_like(subsamples)
Y_sample = np.zeros((ny,nsubsamples))
for i in range(nsubsamples):
    xt_sample = l96model(T=0.2,dt=dt,nx=nx,gamma=gamma,x0=subsamples[:,i])
    XT_sample[:,i] = xt_sample[:,-1]
    Y_sample[:,i] = xt_sample[idx,-1]

y_best_est = l96model(T=0.2,dt=dt,nx=nx,gamma=gamma,x0=mu_post)

axis_data = range(0,nx,2)
fig, ax = plt.subplots()
ax.plot(range(nx),XT_sample,color='#8f99fb',linewidth=0.3,label='_nolegend_')
ax.plot(range(nx),y_best_est[:,-1],'g',linewidth=1.5)
ax.plot(range(nx),linear_best_est,'r',linewidth=1.5)
ax.plot(range(nx),y_true,'b',linewidth=1.5)
ax.errorbar(axis_data,y,yerr=2*np.ones(len(y)),fmt='o',color='k')
ax.legend(['XT MCMC','XT Gauss-Newton','XT0','Ytrue'])
ax.set_title('Uncertainty Quanitification at time T=0.2')
ax.set_xlabel('X Index')
ax.set_ylabel('Values')
plt.savefig('Code/Homework3/MCMC_UQ.png')
plt.show()