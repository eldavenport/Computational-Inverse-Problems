# Homework 2 - linear UQ

import numpy as np
import matplotlib.pyplot as plt
from LSQCost import cost_fxn_resid
from scipy.optimize import least_squares
from L96Model import l96model
from os import getcwd
import numpy.matlib

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
Rsq = U @ np.diag(l**0.5)
l, U = np.linalg.eig(B)
B_sq = U @ np.diag(l**-0.5)
Bsq = U @ np.diag(l**0.5) 
np.save('Code/Homework2/HW2_R_sq.npy',R_sq)
np.save('Code/Homework2/HW2_B_sq.npy',B_sq)
np.save('Code/Homework2/HW2_Rsq.npy',Rsq)
np.save('Code/Homework2/HW2_Bsq.npy',Bsq)

def nonlinsq_l96(y, guess):

    M = lambda x0: l96model(T=0.2,dt=0.01,nx=40,x0=x0,gamma=8)
    resid = lambda x0: cost_fxn_resid(M,R_sq,B_sq,mu,y,x0)
    sol = least_squares(fun=resid, x0=guess, method='lm')
    return sol

# find x* and J at the bets estimate
guess = np.random.normal(0,1,nx)
y = y_true + np.random.normal(0,1,len(y_true))
np.save('Code/Homework2/HW2_y.npy',y)
np.save('Code/Homework2/HW2_x0.npy',x0)
np.save('Code/Homework2/HW2_ytrue.npy',y2[:,-1])
np.save('Code/Homework2/HW2_mu.npy',mu)

solution = nonlinsq_l96(y,guess)
x_star = solution.x 
y_best_est = l96model(T=0.2,dt=dt,nx=nx,gamma=gamma,x0=x_star)
y_best_est = y_best_est[:,-1]
J_x_star = solution.jac

np.save('Code/Homework2/HW2_linearBestEst.npy',y_best_est)
# np.save('Code/Homework2/HW2_x.npy',x_star)
# np.save('Code/Homework2/HW2_J.npy',J_x_star)

# SVD to get the cov matrix (GN approx hessian we want)
U, S, Vh = np.linalg.svd(J_x_star)

# J is 60x40, S is 1x40, and Vh is 40x40 (V should be Vh in the notes)
# this should amount to 40x1 + (40x40)(40x40)(40x1)
Nsamples = 200
sigma_inv = np.diag(1/S)
sample_noises = np.random.normal(0,1,(nx,Nsamples))
samples = numpy.matlib.repmat(x_star[:,np.newaxis],1,Nsamples) + 1/np.sqrt(2) * (Vh.T @ sigma_inv) @ sample_noises

fig, ax = plt.subplots()
ax.plot(range(nx),samples,color='#8f99fb',linewidth=0.3,label='_nolegend_')
ax.plot(range(nx),x0,'b',linewidth=1.5)
ax.plot(range(nx),x_star,'g',linewidth=1.5)
ax.legend(['X0','X*'])
ax.set_title('Uncertainty Quanitification of X*')
ax.set_xlabel('X Index')
ax.set_ylabel('Values')
plt.savefig('Homework2/X_star_UQ.png')
plt.show()
plt.close()

print('running samples forward')
XT_sample = np.zeros_like(samples)
Y_sample = np.zeros((ny,Nsamples))
for i in range(Nsamples):
    xt_sample = l96model(T=0.2,dt=dt,nx=nx,gamma=gamma,x0=samples[:,i])
    XT_sample[:,i] = xt_sample[:,-1]
    Y_sample[:,i] = xt_sample[idx,-1]

axis_data = range(0,nx,2)
fig, ax = plt.subplots()
ax.plot(range(nx),XT_sample,color='#8f99fb',linewidth=0.3,label='_nolegend_')
ax.plot(range(nx),y2[:,-1],'b',linewidth=1.5)
ax.plot(range(nx),y_best_est,'g',linewidth=1.5)
ax.errorbar(axis_data,y,yerr=2*np.ones(len(y)),fmt='o',color='#c41e3a')
ax.legend(['XT0','XT*','Ytrue'])
ax.set_title('Uncertainty Quanitification of X* at time T=0.2')
ax.set_xlabel('X Index')
ax.set_ylabel('Values')
plt.savefig('Homework2/X_star_T2_UQ.png')
plt.show()
