# homework 5 - implementation of harmonic oscillator w Kalman Filter

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

# ----------------------------------------------------------------- Set Params ----------------------------------------------------------------------------------
w = 5
zeta = .01
dt = 0.001
R = 5
T = 100
nx = 2
ny = 2
ntimesteps = np.int32(T/dt)

x0 = np.array([1,2])
print('initial condition: ' + str(x0))

M = np.array([[1, dt],[-(w**2)*dt, (1 - 2*zeta*w*dt)]])
print('M: ' + str(M))

H = np.array([1,0]).T
print('H: ' + str(H))

# ----------------------------------------------------------------- Simulate and Filter ----------------------------------------------------------------------------------
x = np.zeros((nx,ntimesteps))
y = np.zeros((1,ntimesteps))
x_hat = np.zeros((nx,ntimesteps))
Pf = np.zeros((nx,nx,ntimesteps))
Pa = np.zeros((nx,nx,ntimesteps))
RMSE = np.zeros((ntimesteps))
spread = np.zeros((ntimesteps))

# initial conditions
x[:,0] = x0
x_hat[:,0] = x0
Pf[:,:,0] = np.eye(nx)
Pa[:,:,0] = np.eye(nx)
y[0] = 0
for i in range(1,ntimesteps):
    # simulate
    x[:,i] = M @ x[:,i-1]
    y[:,i] = H @ x[:,i] + np.sqrt(R)*np.random.normal(0,1,1)

    #forecast
    xf = M @ x_hat[:,i-1]
    Pf[:,:,i] = M @ Pa[:,:,i-1] @ M.T

    #analysis
    arg = np.array(H @ Pf[:,:,i] @ H.T + R)
    K = Pf[:,:,i] @ H.T * 1/arg

    x_hat[:,i] = xf +  K * (y[:,i] - H @ xf)
    Pa[:,:,i] = (1 - K @ H) * Pf[:,:,i]

    #RMSE (true state and analysis mean) and spread
    spread[i] = np.sqrt(0.5*np.trace(Pa[:,:,i]))
    RMSE[i] = np.linalg.norm(x_hat[:,i]-x[:,i],ord=2)/np.sqrt(nx)

# ----------------------------------------------------------------- Plotting ----------------------------------------------------------------------------------
fig, ax = plt.subplots(nrows=2,constrained_layout=True)
ax[0].plot(range(0,ntimesteps),x[0,:],color='#404be3',linewidth=3)
ax[0].set_title('Position')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Position')

ax[1].plot(range(0,ntimesteps),x[1,:],color='#217b7e',linewidth=3)
ax[1].set_title('Velocity')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Velocity')
plt.savefig('Code/Homework5/simulated_state_zeta01_w5.png',format='png')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8,6),nrows=2,constrained_layout=True)
ax[0].plot(range(0,ntimesteps),x[0,:],color='#404be3',linewidth=3)
ax[0].scatter(range(0,ntimesteps),y,color='#eda323',s=5)
ax[0].legend(['True Position','Data'],loc='upper right')
ax[0].set_title('Position')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Position')

ax[1].plot(range(0,ntimesteps),x[1,:],color='#217b7e',linewidth=3)
ax[1].set_title('Velocity')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Velocity')
plt.savefig('Code/Homework5/simulated_data_zeta01_w5.png',format='png')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8,6),nrows=2,constrained_layout=True)
ax[0].plot(range(0,ntimesteps),x[0,:],color='#404be3',linewidth=3)
ax[0].plot(range(0,ntimesteps),x_hat[0,:],color='#fd4299',linestyle='--',linewidth=3)
ax[0].scatter(range(0,ntimesteps),y,color='#eda323',s=5)
ax[0].legend(['True Position','Estimated Position','Data'],loc='upper right')
ax[0].set_title('Position')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Position')

ax[1].plot(range(0,ntimesteps),x[1,:],color='#217b7e',linewidth=3)
ax[1].plot(range(0,ntimesteps),x_hat[1,:],color='#c06df8',linestyle='--',linewidth=3)
ax[1].legend(['True Velocity','Estimated Velocity'],loc='upper right')
ax[1].set_title('Velocity')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Velocity')
plt.savefig('Code/Homework5/estimated_state_zeta01_w5.png',format='png')
plt.show()
plt.close()

nshort_axis = 1000
fig, ax = plt.subplots(figsize=(8,6),nrows=2,constrained_layout=True)
ax[0].plot(range(0,nshort_axis),x[0,:nshort_axis],color='#404be3',linewidth=3)
ax[0].plot(range(0,nshort_axis),x_hat[0,:nshort_axis],color='#fd4299',linestyle='--',linewidth=3)
ax[0].scatter(range(0,nshort_axis),y[:,:nshort_axis],color='#eda323',s=5)
ax[0].legend(['True Position','Estimated Position','Data'],loc='upper right')
ax[0].set_title('Position')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Position')

ax[1].plot(range(0,nshort_axis),x[1,:nshort_axis],color='#217b7e',linewidth=3)
ax[1].plot(range(0,nshort_axis),x_hat[1,:nshort_axis],color='#c06df8',linestyle='--',linewidth=3)
ax[1].legend(['True Velocity','Estimated Velocity'],loc='lower right')
ax[1].set_title('Velocity')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Velocity')
plt.savefig('Code/Homework5/estimated_state_spinup_zeta01_w5.png',format='png')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8,8),nrows=3,constrained_layout=True)
ax[0].plot(range(0,50000),RMSE[:50000])
ax[0].set_title('RMSE v Time (true state v analysis mean)')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('RMSE')

ax[1].plot(range(0,50000),spread[:50000])
ax[1].set_title('Spread v Time ')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Spread')

ax[2].hist(RMSE[:50000],100,color='#404be3')
ax[2].hist(spread[:50000],100,color='#217b7e')
ax[2].legend(['RMSE','Spread'])
ax[2].set_title('Spread and RMSE')
ax[2].set_xlabel('Magnitude')
ax[2].set_ylabel('Total instances')

plt.savefig('Code/Homework5/RMSE_and_spread_zeta01_w5.png',format='png')
plt.show()
plt.close()