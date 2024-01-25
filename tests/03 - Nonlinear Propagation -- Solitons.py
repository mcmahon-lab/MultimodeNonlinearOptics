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
# # Solitons

# %%
# cd ..

# %%
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fftshift

from nonlinearmedium import Chi3

# %%
# %matplotlib notebook
plt.rcParams["figure.figsize"] = [9, 6]

# %% [markdown]
# # First order soliton
# Stationary solution

# %% [markdown]
# A soliton period is $\zeta=\pi/2$, and is defined as the distance over which a soliton accumulate $\pi/4$ phase from SPM.

# %%
fiber = Chi3(relativeLength=4 * np.pi,
             nlLength=1,
             beta2=-1,
             pulseType=1,
             tPrecision=512, zPrecision=100)

fiber.runPumpSimulation()

# %%
plt.figure()
plt.imshow(np.abs(fftshift(fiber.pumpTime, axes=1)).T, aspect="auto",
           extent=[0, 4 * np.pi, np.max(fiber.tau), np.min(fiber.tau)])
plt.colorbar();
plt.title("Pulse Temporal Field Profile")
plt.ylabel("time")
plt.xlabel("length");

# %%
plt.figure()
plt.imshow(np.real(fftshift(fiber.pumpFreq, axes=1)).T, aspect="auto",
           extent=[0, 4 * np.pi, np.max(fiber.omega), np.min(fiber.omega)])
plt.colorbar();
plt.title("Pulse Spectral Field Profile")
plt.ylabel("angular frequency")
plt.xlabel("length");

# %%
plt.figure()
plt.plot(fftshift(fiber.tau), fftshift(np.abs(fiber.pumpTime[0])), label="Initial")
plt.plot(fftshift(fiber.tau), fftshift(np.abs(fiber.pumpTime[-1])), label="Final")
plt.title("Pulse Temporal Field Profile")
plt.xlabel("time")
plt.ylabel("field")
plt.legend();

# %%
plt.figure()
plt.plot(fftshift(fiber.omega), fftshift(np.abs(fiber.pumpFreq[0])), label="Initial")
plt.plot(fftshift(fiber.omega), fftshift(np.abs(fiber.pumpFreq[-1])), label="Final")
plt.title("Pulse Spectral Field Profile")
plt.xlabel("angular frequency")
plt.ylabel("field")
plt.legend();

# %% [markdown]
# # Higher order solitons
# Periodic solutions, where $N^2=L_D/L_{NL}$.
# The soliton period above defines the spatial periodicity.

# %%
fiber = Chi3(relativeLength=4 * np.pi,
             nlLength=0.25,
             beta2=-1,
             pulseType=1,
             tPrecision=512, zPrecision=100)

fiber.runPumpSimulation()

# %%
plt.figure()
plt.imshow(np.abs(fftshift(fiber.pumpTime, axes=1)).T, aspect="auto",
           extent=[0, 4 * np.pi, np.max(fiber.tau), np.min(fiber.tau)])
plt.colorbar();
plt.title("Pulse Temporal Field Profile")
plt.ylabel("time")
plt.xlabel("length");

# %%
plt.figure()
plt.imshow(np.real(fftshift(fiber.pumpFreq, axes=1)).T, aspect="auto",
           extent=[0, 4 * np.pi, np.max(fiber.omega), np.min(fiber.omega)])
plt.colorbar();
plt.title("Pulse Spectral Field Profile")
plt.ylabel("angular frequency")
plt.xlabel("length");

# %%
fiber = Chi3(relativeLength=4 * np.pi,
             nlLength=1/9,
             beta2=-1,
             pulseType=1,
             tPrecision=512, zPrecision=100)

fiber.runPumpSimulation()

# %%
plt.figure()
plt.imshow(np.abs(fftshift(fiber.pumpTime, axes=1)).T, aspect="auto",
           extent=[0, 4 * np.pi, np.max(fiber.tau), np.min(fiber.tau)])
plt.colorbar();
plt.title("Pulse Temporal Field Profile")
plt.ylabel("time")
plt.xlabel("length");

# %%
plt.figure()
plt.imshow(np.real(fftshift(fiber.pumpFreq, axes=1)).T, aspect="auto",
           extent=[0, 4 * np.pi, np.max(fiber.omega), np.min(fiber.omega)])
plt.colorbar();
plt.title("Pulse Spectral Field Profile")
plt.ylabel("angular frequency")
plt.xlabel("length");
