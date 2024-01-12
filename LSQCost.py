# python file implementing LSQ cost function
import numpy as np
from L96Model import l96model

# this is the regularlized cost function with error defined in the least squares sense
def cost_fxn(M, R_sq, B_sq, mu, y, x0):
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