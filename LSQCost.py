# python file implementing LSQ cost function
import numpy as np
from L96Model import l96model
from numba import njit

# this is the regularlized cost function with error defined in the least squares sense
# @njit
def cost_fxn_resid(M, R_sq, B_sq, mu, y,x0):
    """
    Least Squares Cost Function
    x0 - initial condition
    y - data
    R_sq - matrix sqrt of R (error cov)
    B_sq - matrix  sqrt of B (background cov)
    H - transformation matrix - makes nx,nx model out put ny,nx
    M - model
    mu - regularlization term (background mean)

    """
    # compute F
    est_y = M(x0)
    nx = x0.size
    idx = np.arange(0,nx,2)
    misfit = y - est_y[idx,-1]

    misfitTerm = R_sq @ misfit
    regTerm = B_sq @ (x0 - mu)

    # F = 0.5 * np.linalg.norm(misfitTerm, ord=2) + 0.5 * np.linalg.norm(regTerm, ord=2)
    r = 1/np.sqrt(2) * np.concatenate([misfitTerm,regTerm])
    return r

def cost_fxn_scalar(x0, R_sq, B_sq, mu, y):
    """
    Least Squares Cost Function to be used with emcee 
    x0 - initial condition
    y - data
    R_sq - matrix sqrt of R (error cov)
    B_sq - matrix  sqrt of B (background cov)
    H - transformation matrix - makes nx,nx model out put ny,nx
    M - model
    mu - regularlization term (background mean)

    """
    # compute F
    # est_y = M(x0)
    x0 = x0.T
    est_y = l96model(T=0.2,dt=0.01,nx=40,x0=x0,gamma=8)
    nx = x0.size
    idx = np.arange(0,nx,2)
    misfit = y - est_y[idx,-1]

    misfitTerm = R_sq @ misfit
    regTerm = B_sq @ (x0 - mu)

    # F = 0.5 * np.linalg.norm(misfitTerm, ord=2) + 0.5 * np.linalg.norm(regTerm, ord=2)
    r = 1/np.sqrt(2) * np.concatenate([misfitTerm,regTerm])
    F = -1*(np.dot(r,r.T)) #p(x) is proportional to the exp(-F), for emcee this needs to return the log of the probabilit which is just -F
    # print(F)
    return F