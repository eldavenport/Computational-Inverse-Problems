# homework 3 - nonlinear UQ

import numpy as np
import matplotlib.pyplot as plt
from LSQCost import cost_fxn_scalar
import numpy.matlib
import emcee # import MCMC Hammer solver 
import time
import multiprocessing

multiprocessing.set_start_method("fork")
from multiprocessing import Pool
from multiprocessing import cpu_count
import os

os.environ["OMP_NUM_THREADS"] = "1"
ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

np.random.seed(5)

x_star = np.load('Code/Homework2/HW2_x.npy')
J = np.load('Code/Homework2/HW2_J.npy')
R_sq = np.load('Code/Homework2/HW2_R_sq.npy')
B_sq = np.load('Code/Homework2/HW2_B_sq.npy')
y = np.load('Code/Homework2/HW2_y.npy')

mu = np.load('Code/Homework2/HW2_mu.npy')

Ne = 81
nx = len(mu)
ny = nx/2

# cost_fxn_scalar(M,R_sq,B_sq,mu,y,x0)
# args are constant every time this is run, these are inputs to the cost function (R_sq, B_sq, mu, y)
# initial guess is ne x nx so in our case 50 x 40
# should mu be the linearized UQ mu? or should it be the lorenz system background mu?

# initialize walkers using J from linearized UQ
U, S, Vh = np.linalg.svd(J)

# J is 60x40, S is 1x40, and Vh is 40x40
sigma_inv = np.diag(1/S)
sample_noises = np.random.normal(0,1,(nx,Ne))
p0 = numpy.matlib.repmat(x_star[:,np.newaxis],1,Ne) + 1/np.sqrt(2) * (Vh.T @ sigma_inv) @ sample_noises
p0 = p0.T


nsteps = 50000
with Pool() as pool:
    # Run at staved state for some number of steps
    # intialize sampler
    sampler = emcee.EnsembleSampler(Ne, nx, moves=emcee.moves.StretchMove(), log_prob_fn=cost_fxn_scalar, args=[R_sq, B_sq, mu, y])
    # Burn in
    startTime = time.time()
    state = sampler.run_mcmc(p0, 1000)
    endTime = time.time()
    print(sampler.acceptance_fraction)
    total_time = endTime - startTime
    print("Burn in time {0:.1f} seconds".format(total_time))

    sampler.reset()
    print('burn in done')
    startTime = time.time()
    state = sampler.run_mcmc(state, nsteps)
    endTime = time.time()
    print(sampler.acceptance_fraction)

    # startTime = time.time()
    # state = sampler.run_mcmc(state, nsteps)
    # endTime = time.time()
    # print(sampler.acceptance_fraction)
    # total_time = endTime - startTime
    # print("time {0:.1f} seconds".format(total_time))
    samples = sampler.get_chain(flat=True)

print('done')

total_time = endTime - startTime
print("Total time {0:.1f} seconds".format(total_time))

np.save('Code/Homework3/samples.npy',samples)
np.save('Code/Homework3/alpha.npy',np.mean(sampler.acceptance_fraction))

np.save('Code/Homework3/ACT.npy',np.mean(sampler.get_autocorr_time()))

plt.hist(samples[:, 0], 100, color="k", histtype="step")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([])