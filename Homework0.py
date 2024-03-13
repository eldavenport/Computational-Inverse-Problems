import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from L96Model import l96model
from os import getcwd

np.random.seed(5)
pwd = getcwd()

## 1 & 2 ------------------------------------------------------------------------------
# Write a function that integrates L96 forward with RK4 as the integration scheme
# Run this integrator for 1000 time units 

nx = 40
dt = 0.01
T = 1000
gamma = 8
x_noise = np.random.uniform(0,1,nx)

y1 = l96model(T=1000,dt=dt,nx=nx,gamma=gamma,x0=x_noise)
x0 = y1[:,-1]
print(y1.shape)
# Plot the first three variables for the 1000 time steps
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(y1[0,-1000:], y1[1,-1000:], y1[2, -1000:])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
ax.set_title('L96 Integration, Gamma = 8')
plt.savefig(pwd + '/Homework0/L96.png')
plt.show()

fig, ax = plt.subplots()
pos1 = ax.imshow(y1[:,:10000],vmin=-1*np.max(abs(y1)),vmax=np.max(abs(y1)),cmap='bwr',origin='lower')
im_ratio = y1[:,:10000].shape[0]/(y1[:,:10000].shape[1]/30)
cbar = fig.colorbar(pos1,ax=ax,fraction=0.047*im_ratio)
ax.set_xlabel("Time Step")
ax.set_ylabel("X Index")
ax.set_title('L96 Integration, Gamma = 8')
ax.set_aspect(30)
plt.savefig(pwd + '/Homework0/L96_hovmoller.png')
plt.show()

## 3 ----------------------------------------------------------------------------------
# create synthetic data using the result above (y) as the initial condition
y2 = l96model(T=0.2,dt=dt,nx=nx,gamma=gamma,x0=x0)

# select every other state variable from the last time step
idx = np.arange(0,nx,2)
y2_subset = y2[idx,-1]
y = y2_subset + np.random.normal(0,1,len(y2_subset))

## 4 ----------------------------------------------------------------------------------
# Make background cov matrix B using a long simulation (several thousand time units)
b = l96model(T=5000,dt=dt,nx=nx,gamma=gamma,x0=x0)
B = np.cov(b)
mu = b.mean(axis=1) # take the time mean 

fig, ax = plt.subplots()
pos1 = ax.imshow(B,cmap='PuOr',vmin=-1*np.max(abs(B)),vmax=np.max(abs(B)))
im_ratio = B.shape[0]/B.shape[1]
cbar2 = fig.colorbar(pos1,ax=ax,fraction=0.047*im_ratio)
ax.set_xlabel('X index')
ax.set_ylabel('Y index')
ax.set_title('Background covariance, B')
plt.savefig(pwd + '/Homework0/B.png')