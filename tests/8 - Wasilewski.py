# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Comparisons to theoretical predictions of Green's Functions
# Pulsed squeezed light: Simultaneous squeezing of multiple modes
# (Wasilewski, Lvovsky, Banaszek, Radzewicz)

# %%
# cd ..

# %%
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

try:
    from nonlinearmedium import Chi2PDC
    print("using C++ implementation")
except:
    from NonlinearMedium import Chi2PDC
    print("using Python implementation")

from NonlinearHelper import *

# %%
# %matplotlib notebook
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 6]

# %% [markdown]
# The paper makes predictions about the Green's function in the weak generation regime. We consider the case without dispersion at either frequencies.

# %%
length = 0.05
nFreqs = 512
crys = Chi2PDC(relativeLength=length,
               nlLength=1,
               beta2=0,
               beta2s=0,
               pulseType=0,
               tPrecision=nFreqs, zPrecision=20000)

crys.runPumpSimulation()

# %% [markdown]
# # Expected vs calculated Green's function
# In the weak generation regime expect the Green Function to be of the form
# $$
# S(\omega, \omega') = E(\omega + \omega') L / L_{NL}
# $$

# %%
weakC, weakS = crys.computeGreensFunction(runPump=False)

# %%
fig = plt.figure()
crossSection = np.abs(weakS[:, nFreqs // 2])
plt.plot(crossSection, label="Simulated")

prediction = np.abs(fftshift(crys.pumpFreq[0] * length))
plt.plot(prediction / nFreqs, label="Predicted")

plt.xlabel("$\omega$")
plt.title("$|S(\omega,\omega')|$ Green's function cross-section at $\omega' = 0$")
plt.legend();

# %% [markdown]
# # Entire Green's function

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(np.abs(fftshift(weakC)), origin='lower')
plt.title("$|C(\omega, \omega')|$")
plt.xlabel("$\omega'$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar()
ax = fig.add_subplot(1, 2, 2)
plt.imshow(np.abs(fftshift(weakS)), origin='lower')
plt.title("$|S(\omega, \omega')|$")
plt.xlabel("$\omega'$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar();

# %% [markdown]
# # Also test time domain Green's function
# Expect
# $$
# S(t, t') = \delta(t - t') E(t') L / L_{NL}
# $$

# %%
weakCT, weakST = crys.computeGreensFunction(runPump=False, inTimeDomain=True)

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(np.abs(weakCT), origin='lower')
plt.title("$|C(t,t')|$")
plt.xlabel("$t'$")
plt.ylabel("$t$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar()
ax = fig.add_subplot(1, 2, 2)
plt.imshow(np.abs(weakST), origin='lower')
plt.title("$|S(t,t')|$")
plt.xlabel("$t'$")
plt.ylabel("$t$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar();

# %%
fig = plt.figure()
crossSection = np.abs(np.diag(weakST))
plt.plot(crossSection, label="Simulated")

prediction = np.abs(fftshift(crys.pumpTime[0] * length))
plt.plot(prediction, label="Predicted")

plt.xlabel("t")
plt.title("$|S(t,t')|$ Green's function diagonal $t=t'$")
plt.legend();
