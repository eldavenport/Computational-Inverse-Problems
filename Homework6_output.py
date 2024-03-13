# homework 6 processing

import numpy as np
import matplotlib.pyplot as plt

x_true = np.load('Homework6/x_true.npy')
RMSE = np.load('Homework6/RMSE.npy')
spread = np.load('Homework6/spread.npy')
y = np.load('Homework6/y.npy')
mu_a = np.load('Homework6/mu_a.npy')

T=300
dt=0.2

print(np.nanmean(RMSE[100:]))
fig, ax = plt.subplots(nrows=2,constrained_layout=True)
ax[0].plot(RMSE)
ax[0].set_title('RMSE')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('RMSE')
# ax[0].set_ylim(0,10)
ax[0].set_xlim(0,T/dt)
ax[0].hlines(y=np.mean(RMSE[100:]),xmin=0,xmax=T/dt,color='r')

print(np.nanmean(spread[100:]))
ax[1].plot(spread)
ax[1].set_title('Spread')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Spread')
# ax[1].set_ylim(0,1.5)
ax[1].set_xlim(0,T/dt)
ax[1].hlines(y=np.nanmean(spread[100:]),xmin=0,xmax=T/dt,color='r')
plt.savefig('Homework6/RMSE_Spread.png')
plt.show()
plt.close()

vmin=-12
vmax=12
fig, ax = plt.subplots(figsize=(6,7),nrows=3,constrained_layout=True)
pos = ax[0].imshow(x_true,cmap='bwr',vmin=vmin,vmax=vmax)
ax[0].set_title('True System')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('X[i]')
ax[0].set_aspect(6)
im_ratio = x_true.shape[0]/(x_true.shape[1]/6)
cbar = fig.colorbar(pos,ax=ax[0],fraction=0.047*im_ratio)

pos = ax[1].imshow(y,cmap='bwr',vmin=vmin,vmax=vmax)
ax[1].set_title('Data')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Y[i]')
ax[1].set_aspect(12)
im_ratio = y.shape[0]/(y.shape[1]/12)
cbar = fig.colorbar(pos,ax=ax[1],fraction=0.047*im_ratio)

pos = ax[2].imshow(mu_a,cmap='bwr',vmin=vmin,vmax=vmax)
ax[2].set_title('Est State')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('$mu_f$[i]')
ax[2].set_aspect(6)
im_ratio = x_true.shape[0]/(x_true.shape[1]/6)
cbar = fig.colorbar(pos,ax=ax[2],fraction=0.047*im_ratio)
plt.savefig('Homework6/OutputImg.png')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(6,7),nrows=3,constrained_layout=True)
pos = ax[0].imshow(x_true,cmap='bwr',vmin=vmin,vmax=vmax)
ax[0].set_title('True System')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('x')
ax[0].set_aspect(6)
im_ratio = x_true.shape[0]/(x_true.shape[1]/6)
cbar = fig.colorbar(pos,ax=ax[0],fraction=0.047*im_ratio)

pos = ax[1].imshow(mu_a,cmap='bwr',vmin=vmin,vmax=vmax)
ax[1].set_title('Est State')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('$mu_f$')
ax[1].set_aspect(6)
im_ratio = x_true.shape[0]/(x_true.shape[1]/6)
cbar = fig.colorbar(pos,ax=ax[1],fraction=0.047*im_ratio)

vmin=-10
vmax=10
pos = ax[2].imshow(x_true - mu_a,cmap='bwr',vmin=vmin,vmax=vmax)
ax[2].set_title('True State - Est State')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('x - $mu_f$')
ax[2].set_aspect(6)
im_ratio = x_true.shape[0]/(x_true.shape[1]/6)
cbar = fig.colorbar(pos,ax=ax[2],fraction=0.047*im_ratio)
plt.savefig('Homework6/TrueEstDifference.png')
plt.show()
plt.close()