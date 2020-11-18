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

# %%
# cd ..

# %%
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

try:
    from nonlinearmedium import Chi2SFG
except:
    from NonlinearMedium import Chi2SFG

from NonlinearHelper import *

# %%
# %matplotlib notebook
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 6]

# %% [markdown]
# # Check the effect of Quasi Phase Matching on a $\chi^{(2)}$ system that is not phase matched

# %% [markdown]
# Simulate with different step sizes, for 2 configurations:
#
# a) Poled (flipping the sign of the nonlinear term in each domain)
#
# b) Rotating frame ($\Delta k \rightarrow 0$, $d_{eff} \rightarrow 2 d / \pi$)
#
# For simplicity simulate no dispersion.

# %%
zSteps = [50, 100, 200, 500, 1000, 5000]
zStepsPM = [50, 100, 200]
nt = 2**8

# %%
baseParams = {
    "relativeLength": 3,
    "nlLength": 1,
    "nlLengthOrig": 1,
    "dispLength": np.inf,
    "beta2": 0,
    "beta2s": 0,
    "beta2o": 0,
    "customPump": np.ones(nt),
    "tPrecision": nt,
}

# %% [markdown]
# Phase Mismatch and Domain Poling

# %%
diffBeta0 = 150

# %%
polPeriod = 2 * np.pi / abs(diffBeta0)
nDomains  = 2 * baseParams["relativeLength"] / polPeriod
poling = np.ones(int(nDomains) + int(np.ceil(nDomains % 1)))
poling[-1] = nDomains % 1

# %%
tests = [None] * (len(zStepsPM) + len(zSteps))

for i, z in enumerate(zStepsPM):
    tests[i] = Chi2SFG(**{**baseParams, "zPrecision": z,
                          "relativeLength": baseParams["relativeLength"] / (np.pi / 2)})

for i, z in enumerate(zSteps):
    tests[i+len(zStepsPM)] = Chi2SFG(**{**baseParams, "zPrecision": z,
                                        "diffBeta0": diffBeta0, "poling": poling})

# %% [markdown]
# Effective Poling Profile for each simulation

# %%
for crystal in tests:
  fig = plt.figure()
  ax = fig.add_subplot(2, 1, 1)

  spatialFreq = np.abs(fftshift(fft(crystal.poling)))
  plt.plot(2 * np.pi / baseParams["relativeLength"] * np.arange(-crystal.pumpTime.shape[0] / 2, crystal.pumpTime.shape[0] / 2),
           spatialFreq / np.max(spatialFreq), label=r"$k_{QPM}$")
  plt.plot([-diffBeta0, -diffBeta0, diffBeta0, diffBeta0],
           [1, 0, 0, 1], label=r"$\Delta\Beta_0$")
  plt.xlabel(r"$k_z$")
  plt.ylabel(r"$\chi^{(2)}_{eff}(k_z) ~/~ \chi^{(2)}$")
  plt.title("Poling frequency profiles")
  fig.legend()

  ax = fig.add_subplot(2, 1, 2)
  plt.plot(baseParams["relativeLength"] * np.arange(-crystal.pumpTime.shape[0] / 2, crystal.pumpTime.shape[0] / 2) / nt,
           crystal.poling, "-o")
  plt.xlabel(r"$z$")
  plt.ylabel(r"$\chi^{(2)}_{eff}(z) ~/~ \chi^{(2)}$");

# %%
for test in tests:
  test.runPumpSimulation()
  test.runSignalSimulation(1j * np.exp(-test.tau**2), inTimeDomain=True)

# %%
fig = plt.figure()
# plt.plot(np.linspace(0, baseParams["relativeLength"], tests[-1].pumpTime.shape[0]),
#          (tests[-1].poling + 1) * 0.5)

for z, test in zip(zSteps, tests[:len(zStepsPM)]):
    plt.plot(np.linspace(0, baseParams["relativeLength"], test.pumpTime.shape[0]),
         np.abs(test.signalTimes(1)[:, 0]), label="Phase Matched ({:d} Steps / L_NL)".format(z))

for z, test in zip(zSteps, tests[len(zStepsPM):]):
  plt.plot(np.linspace(0, baseParams["relativeLength"], test.pumpTime.shape[0]),
           np.abs(test.signalTimes(1)[:, 0]), label="{:d} Steps / L_NL".format(z))

plt.xlabel("z")
plt.ylabel("Conversion")
plt.title("SFG Solution Comparison of Simulation Parameters")

plt.legend();

# %% [markdown]
# The discontinuity and high frequency of the domain poling structure significantly increase the resolution needed to obtain the correct solution compared to the rotating frame simulation.
