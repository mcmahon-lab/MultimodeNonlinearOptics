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

from numpy.fft import fft, ifft, fftshift, ifftshift

try:
    from nonlinearmedium import Chi3, Chi2PDC
    print("using C++ implementation")
except:
    from NonlinearMedium import Chi3, Chi2PDC
    print("using Python implementation")

# %%
# %matplotlib notebook
plt.rcParams['figure.figsize'] = [9, 6]

# %%
fiber = Chi3(relativeLength=5,
             nlLength=np.inf,
             dispLength=1,
             beta2=1,
             beta3=1,
             pulseType=0,
             tPrecision=2048, zPrecision=200, tMax=45)

fiber.runPumpSimulation()

# %%
plt.figure()
plt.imshow(np.abs(fftshift(fiber.pumpTime,axes=1)).T, aspect="auto",
           extent=[0, 4 * np.pi, np.min(fiber.tau), np.max(fiber.tau)])
plt.title("Pump Field Profile")
plt.ylabel("time")
plt.xlabel("length");

# %%
plt.figure()
plt.plot(fiber.tau, np.abs(fiber.pumpTime[0])**2, label="Input")
plt.plot(fiber.tau, np.abs(fiber.pumpTime[-1])**2, label="Output")
plt.title("Pump Field Profile")
plt.xlabel("time")
plt.ylabel("field")
plt.legend();
