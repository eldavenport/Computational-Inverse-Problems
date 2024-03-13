# Homework 0
import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# building numerical model for Lorenz 96
from numba import njit

def l96model(T=1000, dt=0.01, nx=1, x0=1, gamma=8):
    # dXi/dt = -Xi + (Xi+1 - Xi-2)Xi-1 + gamma
    # numerically integrate from time 0 to time T using RK4
    # takes initial state input (nx x 1) and outputs the state at time T

    # model returns dX/dt
    def L96(t, x):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        dx_dt = np.zeros(nx)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(nx):
            dx_dt[i] = (x[(i + 1) % nx] - x[i - 2]) * x[i - 1] - x[i] + gamma
        
        return dx_dt

    t = np.arange(0,T+dt,dt)
    soln = solve_ivp(L96, (0, T), x0, t_eval=t)
    return soln.y
