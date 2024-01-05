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
# # Dispersion

# %%
# cd ..

# %%
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import ifft, fftshift

from nonlinearmedium import Chi2PDC, Chi3

# %%
# %matplotlib notebook
plt.rcParams["figure.figsize"] = [9, 6]

# %% [markdown]
# # Qualitative test

# %% [markdown]
# Test that a positive $\beta_1$ moves a pulse backwards in time.
# Test that a positive (normal) group velocity chirps the pulse such that higher frequencies lag and lower frequencies lead.

# %%
fiber = Chi2PDC(relativeLength=5,
                nlLength=np.inf,
                beta2=1,
                beta2s=-1,
                pulseType=0,
                beta1=4,
                beta1s=-4,
                tPrecision=2048, zPrecision=100, tMax=50)

fiber.runPumpSimulation()

# %%
plt.figure()
plt.imshow(np.abs(fftshift(fiber.pumpTime, axes=1)).T, aspect="auto",
           extent=[0, 5, np.max(fiber.tau), np.min(fiber.tau)])
plt.colorbar();
plt.title("Pump Temporal Field Profile")
plt.ylabel("time")
plt.xlabel("length");

# %% [markdown]
# To visualize the chirp modulate by a complex exponential

# %%
plt.figure()
plt.plot(fftshift(fiber.tau), fftshift(np.real(fiber.pumpTime[-1] * np.exp(-4j * fiber.tau))))
plt.title("Pump Temporal Field Profile")
plt.ylabel("Field")
plt.xlabel("Time");

# %% [markdown]
# Same for signal, but flip the sign of two values

# %%
fiber.runSignalSimulation(np.exp(-0.5 * fiber.tau**2))

# %%
plt.figure()
plt.imshow(np.abs(fftshift(fiber.signalTime, axes=1)).T, aspect="auto",
           extent=[0, 5, np.max(fiber.tau), np.min(fiber.tau)])
plt.colorbar();
plt.title("Pump Temporal Field Profile")
plt.ylabel("time")
plt.xlabel("length");

# %%
plt.figure()
plt.plot(fftshift(fiber.tau), fftshift(np.real(fiber.signalTime[-1] * np.exp(4j * fiber.tau))))
plt.title("Pump Temporal Field Profile")
plt.ylabel("Field")
plt.xlabel("Time");

# %% [markdown]
# # Detuning

# %% [markdown]
# Now test the group velocity dispersion by detuning the pulse.
# With normal dispersion, we expect higher frequencies to propagate slower and vice versa.

# %%
fiber = Chi3(relativeLength=5,
             nlLength=np.inf,
             beta2=1,
             pulseType=0,
             tPrecision=2048, zPrecision=100, tMax=50)

detuned = np.exp(-0.5 * (fiber.omega - 5)**2) * np.sqrt(2048)
fiber.runSignalSimulation(ifft(detuned))

# %%
plt.figure()
plt.plot(fftshift(fiber.omega), fftshift(detuned))
plt.title("Spectral Field Profile")
plt.xlabel("angular frequency")
plt.ylabel("field")

# %%
plt.figure()
plt.imshow(np.abs(fftshift(fiber.signalTime, axes=1)).T, aspect="auto",
           extent=[0, 5, np.max(fiber.tau), np.min(fiber.tau)])
plt.colorbar();
plt.title("Pump Temporal Field Profile")
plt.ylabel("time")
plt.xlabel("length");

# %%
plt.figure()
plt.plot(fftshift(fiber.tau), fftshift(np.real(fiber.signalTime[-1] * np.exp(-1j * fiber.tau))))
plt.title("Pump Temporal Field Profile")
plt.ylabel("Field")
plt.xlabel("Time");

# %% [markdown]
# # Dispersion Length
# The temporal profile of a Gaussian pulse should widen as it propagates. The width depends on the propagation length and the dispersion length.
# From Agrawal, Nonlinear Fiber Optics Eq 3.2.11 p.55, $T(z) = T_0 (1 + (z/L_d)^2)^{1/2}$.

# %%
fiber = Chi3(relativeLength=5,
             nlLength=np.inf,
             beta2=1,
             pulseType=0,
             tPrecision=2048, zPrecision=100, tMax=50)

fiber.runPumpSimulation()

# %%
plt.figure()
plt.imshow(np.abs(fftshift(fiber.pumpTime, axes=1)).T, aspect="auto",
           extent=[0, 5, np.max(fiber.tau), np.min(fiber.tau)])
plt.colorbar();
plt.title("Pump Temporal Field Profile")
plt.ylabel("time")
plt.xlabel("length");

# %%
plt.figure()
for i in range(6):
    profile = np.abs(fiber.pumpTime[int((fiber.pumpTime.shape[0]-1) * i / 5)])
    plt.plot(fftshift(fiber.tau), fftshift(profile / np.max(profile)), label="z="+str(i))
for i in range(5,-1, -1):
    plt.plot(np.sqrt(1 + i**2) * np.array([-1, 1]), [np.exp(-0.5), np.exp(-0.5)])
plt.title("Pump Temporal Field Profiles")
plt.ylabel("Field")
plt.xlabel("Time")
plt.legend();
