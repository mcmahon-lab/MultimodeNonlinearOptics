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
# # Tests of third order dispersion
# Comparing to Agrawal, Nonlinear Fiber Optics, Figure 3.6, p.64

# %%
# cd ..

# %%
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fftshift

from nonlinearmedium import Chi3, Chi2PDC

# %%
# %matplotlib notebook
plt.rcParams["figure.figsize"] = [9, 6]

# %%
fiber1 = Chi3(relativeLength=5,
              nlLength=np.inf,
              beta2=1,
              beta3=1,
              pulseType=0,
              tPrecision=2048, zPrecision=200, tMax=45)

fiber1.runPumpSimulation()

# %%
plt.figure()
plt.imshow(np.abs(fftshift(fiber1.pumpTime,axes=1)).T, aspect="auto",
           extent=[0, 5, np.min(fiber1.tau), np.max(fiber1.tau)])
plt.title("Pump Field Profile")
plt.ylabel("time")
plt.xlabel("length");

# %%
fiber2 = Chi3(relativeLength=5,
              nlLength=np.inf,
              beta2=0,
              beta3=1,
              pulseType=0,
              tPrecision=2048, zPrecision=200, tMax=45)

fiber2.runPumpSimulation()

# %%
plt.figure()
plt.imshow(np.abs(fftshift(fiber2.pumpTime,axes=1)).T, aspect="auto",
           extent=[0, 5, np.min(fiber2.tau), np.max(fiber2.tau)])
plt.title("Pump Field Profile")
plt.ylabel("time")
plt.xlabel("length");

# %%
plt.figure()
plt.plot(fftshift(fiber1.tau), fftshift(np.abs(fiber1.pumpTime[0])**2), label=r"$z=0$")
plt.plot(fftshift(fiber1.tau), fftshift(np.abs(fiber1.pumpTime[-1])**2), label=r"$L_D = L_D'$")
plt.plot(fftshift(fiber2.tau), fftshift(np.abs(fiber2.pumpTime[-1])**2), label=r"$\beta_2 = 0$")
plt.title("Pulse Profile")
plt.xlabel("Time")
plt.ylabel("Intensity")
plt.xlim(-6, 12)
plt.legend();
