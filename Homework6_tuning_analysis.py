# analyze tuning results 

import numpy as np
import matplotlib.pyplot as plt

alpha = np.array([0.05,0.075,0.1,0.125,0.15])
lengthscale = np.array([4,6,8,10])

RMSE = np.load('Homework6/Tuning/RMSE_tuning.npy')
spread = np.load('Homework6/Tuning/spread_tuning.npy')

RMSE = RMSE[:len(alpha),:len(lengthscale),100:]
spread = spread[:len(alpha),:len(lengthscale),100:] # get rid of spin up

T=200
dt=0.2

fig, ax = plt.subplots(figsize=(10,8),nrows=2,ncols=5,constrained_layout=True)
ax[0,0].plot(RMSE[0,:,:].T)
ax[0,0].set_title('RMSE, alpha = '+str(alpha[0]))
ax[0,0].legend(['l = 4','l = 6','l = 8','l = 10'])
ax[0,0].set_xlabel('Time')
ax[0,0].set_ylabel('RMSE')
ax[0,0].set_ylim(0,10)
ax[0,0].set_xlim(0,T/dt-100)
# ax[0,0].hlines(y=np.mean(RMSE[0,:,:]),xmin=0,xmax=T/dt,color='r')

ax[1,0].plot(spread[0,:,:].T)
ax[1,0].set_title('Spread, alpha = '+str(alpha[0]))
ax[1,0].legend(['l = 4','l = 6','l = 8','l = 10'])
ax[1,0].set_xlabel('Time')
ax[1,0].set_ylabel('Spread')
ax[1,0].set_ylim(0,10)
ax[1,0].set_xlim(0,T/dt-100)
# ax[1,0].hlines(y=np.nanmean(spread[0,:,:]),xmin=0,xmax=T/dt,color='r')

ax[0,1].plot(RMSE[1,:,:].T)
ax[0,1].set_title('RMSE, alpha = '+str(alpha[1]))
ax[0,1].legend(['l = 4','l = 6','l = 8','l = 10'])
ax[0,1].set_xlabel('Time')
ax[0,1].set_ylabel('RMSE')
ax[0,1].set_ylim(0,10)
ax[0,1].set_xlim(0,T/dt-100)
# ax[0,1].hlines(y=np.mean(RMSE[1,:,:]),xmin=0,xmax=T/dt,color='r')

ax[1,1].plot(spread[1,:,:].T)
ax[1,1].set_title('Spread, alpha = '+str(alpha[1]))
ax[1,1].legend(['l = 4','l = 6','l = 8','l = 10'])
ax[1,1].set_xlabel('Time')
ax[1,1].set_ylabel('Spread')
ax[1,1].set_ylim(0,10)
ax[1,1].set_xlim(0,T/dt-100)
# ax[1,1].hlines(y=np.nanmean(spread[1,:,:]),xmin=0,xmax=T/dt,color='r')

ax[0,2].plot(RMSE[2,:,:].T)
ax[0,2].set_title('RMSE, alpha = '+str(alpha[2]))
ax[0,2].legend(['l = 4','l = 6','l = 8','l = 10'])
ax[0,2].set_xlabel('Time')
ax[0,2].set_ylabel('RMSE')
ax[0,2].set_ylim(0,10)
ax[0,2].set_xlim(0,T/dt-100)
# ax[0,2].hlines(y=np.mean(RMSE[2,:,:]),xmin=0,xmax=T/dt,color='r')

ax[1,2].plot(spread[2,:,:].T)
ax[1,2].set_title('Spread, alpha = '+str(alpha[2]))
ax[1,2].legend(['l = 4','l = 6','l = 8','l = 10'])
ax[1,2].set_xlabel('Time')
ax[1,2].set_ylabel('Spread')
ax[1,2].set_ylim(0,10)
ax[1,2].set_xlim(0,T/dt-100)
# ax[1,2].hlines(y=np.nanmean(spread[2,:,:]),xmin=0,xmax=T/dt,color='r')

ax[0,3].plot(RMSE[3,:,:].T)
ax[0,3].set_title('RMSE, alpha = '+str(alpha[3]))
ax[0,3].legend(['l = 4','l = 6','l = 8','l = 10'])
ax[0,3].set_xlabel('Time')
ax[0,3].set_ylabel('RMSE')
ax[0,3].set_ylim(0,10)
ax[0,3].set_xlim(0,T/dt-100)
# ax[0,2].hlines(y=np.mean(RMSE[2,:,:]),xmin=0,xmax=T/dt,color='r')

ax[1,3].plot(spread[3,:,:].T)
ax[1,3].set_title('Spread, alpha = '+str(alpha[3]))
ax[1,3].legend(['l = 4','l = 6','l = 8','l = 10'])
ax[1,3].set_xlabel('Time')
ax[1,3].set_ylabel('Spread')
ax[1,3].set_ylim(0,10)
ax[1,3].set_xlim(0,T/dt-100)

ax[0,4].plot(RMSE[4,:,:].T)
ax[0,4].set_title('RMSE, alpha = '+str(alpha[4]))
ax[0,4].legend(['l = 4','l = 6','l = 8','l = 10'])
ax[0,4].set_xlabel('Time')
ax[0,4].set_ylabel('RMSE')
ax[0,4].set_ylim(0,10)
ax[0,4].set_xlim(0,T/dt-100)
# ax[0,2].hlines(y=np.mean(RMSE[2,:,:]),xmin=0,xmax=T/dt,color='r')

ax[1,4].plot(spread[4,:,:].T)
ax[1,4].set_title('Spread, alpha = '+str(alpha[4]))
ax[1,4].legend(['l = 4','l = 6','l = 8','l = 10'])
ax[1,4].set_xlabel('Time')
ax[1,4].set_ylabel('Spread')
ax[1,4].set_ylim(0,10)
ax[1,4].set_xlim(0,T/dt-100)

plt.savefig('Homework6/RMSE_Spread_tuning_byalpha.png')
plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(17,5),nrows=2,ncols=4,constrained_layout=True)
ax[0,0].plot(RMSE[:,0,:].T)
ax[0,0].set_title('RMSE, l = '+str(lengthscale[0]))
ax[0,0].legend(['a = 0.05','a = 0.075','a = 0.1','a = 0.125','a = 0.15'])
ax[0,0].set_xlabel('Time')
ax[0,0].set_ylabel('RMSE')
ax[0,0].set_ylim(0,10)
ax[0,0].set_xlim(0,T/dt-100)
# ax[0,0].hlines(y=np.mean(RMSE[0,:,:]),xmin=0,xmax=T/dt,color='r')

ax[1,0].plot(spread[:,0,:].T)
ax[1,0].set_title('Spread, l = '+str(lengthscale[0]))
ax[1,0].legend(['a = 0.05','a = 0.075','a = 0.1','a = 0.125','a = 0.15'])
ax[1,0].set_xlabel('Time')
ax[1,0].set_ylabel('Spread')
ax[1,0].set_ylim(0,10)
ax[1,0].set_xlim(0,T/dt-100)
# ax[1,0].hlines(y=np.nanmean(spread[0,:,:]),xmin=0,xmax=T/dt,color='r')

ax[0,1].plot(RMSE[:,1,:].T)
ax[0,1].set_title('RMSE, l = '+str(lengthscale[1]))
ax[0,1].legend(['a = 0.05','a = 0.075','a = 0.1','a = 0.125','a = 0.15'])
ax[0,1].set_xlabel('Time')
ax[0,1].set_ylabel('RMSE')
ax[0,1].set_ylim(0,10)
ax[0,1].set_xlim(0,T/dt-100)
# ax[0,1].hlines(y=np.mean(RMSE[1,:,:]),xmin=0,xmax=T/dt,color='r')

ax[1,1].plot(spread[:,1,:].T)
ax[1,1].set_title('Spread, l = '+str(lengthscale[1]))
ax[1,1].legend(['a = 0.05','a = 0.075','a = 0.1','a = 0.125','a = 0.15'])
ax[1,1].set_xlabel('Time')
ax[1,1].set_ylabel('Spread')
ax[1,1].set_ylim(0,10)
ax[1,1].set_xlim(0,T/dt-100)
# ax[1,1].hlines(y=np.nanmean(spread[1,:,:]),xmin=0,xmax=T/dt,color='r')

ax[0,2].plot(RMSE[:,2,:].T)
ax[0,2].set_title('RMSE, l = '+str(lengthscale[2]))
ax[0,2].legend(['a = 0.05','a = 0.075','a = 0.1','a = 0.125','a = 0.15'])
ax[0,2].set_xlabel('Time')
ax[0,2].set_ylabel('RMSE')
ax[0,2].set_ylim(0,10)
ax[0,2].set_xlim(0,T/dt-100)
# ax[0,2].hlines(y=np.mean(RMSE[2,:,:]),xmin=0,xmax=T/dt,color='r')

ax[1,2].plot(spread[:,2,:].T)
ax[1,2].set_title('Spread, l = '+str(lengthscale[2]))
ax[1,2].legend(['a = 0.05','a = 0.075','a = 0.1','a = 0.125','a = 0.15'])
ax[1,2].set_xlabel('Time')
ax[1,2].set_ylabel('Spread')
ax[1,2].set_ylim(0,10)
ax[1,2].set_xlim(0,T/dt-100)
# ax[1,2].hlines(y=np.nanmean(spread[2,:,:]),xmin=0,xmax=T/dt,color='r')

ax[0,3].plot(RMSE[:,3,:].T)
ax[0,3].set_title('RMSE, l = '+str(lengthscale[3]))
ax[0,3].legend(['a = 0.05','a = 0.075','a = 0.1','a = 0.125','a = 0.15'])
ax[0,3].set_xlabel('Time')
ax[0,3].set_ylabel('RMSE')
ax[0,3].set_ylim(0,10)
ax[0,3].set_xlim(0,T/dt-100)
# ax[0,2].hlines(y=np.mean(RMSE[2,:,:]),xmin=0,xmax=T/dt,color='r')

ax[1,3].plot(spread[:,3,:].T)
ax[1,3].set_title('Spread, l = '+str(lengthscale[3]))
ax[1,3].legend(['a = 0.05','a = 0.075','a = 0.1','a = 0.125','a = 0.15'])
ax[1,3].set_xlabel('Time')
ax[1,3].set_ylabel('Spread')
ax[1,3].set_ylim(0,10)
ax[1,3].set_xlim(0,T/dt-100)

plt.savefig('Homework6/RMSE_Spread_tuning_byLengthscale.png')
plt.show()
plt.close()

RMSE_mean = np.zeros((len(alpha),len(lengthscale)))
Spread_mean = np.zeros((len(alpha),len(lengthscale)))
RMSE_std = np.zeros((len(alpha),len(lengthscale)))

for a in range(0,len(alpha)):
    for b in range(0,len(lengthscale)):
        RMSE_mean[a,b] = np.mean(RMSE[a,b,:])
        Spread_mean[a,b] = np.mean(spread[a,b,:])
        RMSE_std[a,b] = np.nanstd(RMSE[a,b,:])

diff = RMSE_mean - Spread_mean

fig,ax = plt.subplots(figsize=(10,10),ncols=2,nrows=2)
pos = ax[0,0].imshow(RMSE_mean,cmap='viridis',origin='lower',vmin=0,vmax=2.5)
im_ratio = RMSE_mean.shape[0]/(RMSE_mean.shape[1])
cbar = fig.colorbar(pos,ax=ax[0,0],fraction=0.047*im_ratio)
ax[0,0].set_title('RMSE')
ax[0,0].set_ylabel('alpha')
ax[0,0].set_yticks(range(0,len(alpha)))
ax[0,0].set_yticklabels(alpha.astype(str))
ax[0,0].set_xlabel('length scale')
ax[0,0].set_xticks(range(0,len(lengthscale)))
ax[0,0].set_xticklabels(lengthscale.astype(str))

pos = ax[0,1].imshow(Spread_mean,cmap='viridis',origin='lower',vmin=0,vmax=2.5)
im_ratio = Spread_mean.shape[0]/(Spread_mean.shape[1])
cbar = fig.colorbar(pos,ax=ax[0,1],fraction=0.047*im_ratio)
ax[0,1].set_title('Spread')
ax[0,1].set_ylabel('alpha')
ax[0,1].set_yticks(range(0,len(alpha)))
ax[0,1].set_yticklabels(alpha.astype(str))
ax[0,1].set_xlabel('length scale')
ax[0,1].set_xticks(range(0,len(lengthscale)))
ax[0,1].set_xticklabels(lengthscale.astype(str))

pos = ax[1,0].imshow(diff,cmap='bwr',vmin=-2,vmax=2,origin='lower')
im_ratio = diff.shape[0]/(diff.shape[1])
cbar = fig.colorbar(pos,ax=ax[1,0],fraction=0.047*im_ratio)
ax[1,0].set_title('RMSE - Spread')
ax[1,0].set_ylabel('alpha')
ax[1,0].set_yticks(range(0,len(alpha)))
ax[1,0].set_yticklabels(alpha.astype(str))
ax[1,0].set_xlabel('length scale')
ax[1,0].set_xticks(range(0,len(lengthscale)))
ax[1,0].set_xticklabels(lengthscale.astype(str))

pos = ax[1,1].imshow(RMSE_std,cmap='viridis',origin='lower')
im_ratio = RMSE_std.shape[0]/(RMSE_std.shape[1])
cbar = fig.colorbar(pos,ax=ax[1,1],fraction=0.047*im_ratio)
ax[1,1].set_title('RMSE std dev')
ax[1,1].set_ylabel('alpha')
ax[1,1].set_yticks(range(0,len(alpha)))
ax[1,1].set_yticklabels(alpha.astype(str))
ax[1,1].set_xlabel('length scale')
ax[1,1].set_xticks(range(0,len(lengthscale)))
ax[1,1].set_xticklabels(lengthscale.astype(str))

plt.savefig('Homework6/Tuning_Matrix.png')
plt.show()
plt.close()