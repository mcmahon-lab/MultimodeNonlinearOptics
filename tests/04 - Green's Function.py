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
# # Green's Function

# %% [markdown]
# Obtain the Green's function for the propagation of the signal. Using soliton as a pump and simulating a $\chi (3)$ quantum signal.

# %%
# cd ..

# %%
import numpy as np

from numpy.fft import fft, ifft, fftshift

from nonlinearmedium import Chi3

# %%
# %matplotlib notebook
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 6]

# %% [markdown]
# Using a soliton as a pump

# %%
nFreqs = 512
fiberS = Chi3(relativeLength=np.pi / 2,
              nlLength=1,
              beta2=-1,
              pulseType=1,
              tPrecision=nFreqs, zPrecision=100)

C, S = fiberS.computeGreensFunction(nThreads=4)

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(np.abs(C), origin='lower', aspect="auto")
plt.title("$|C(\omega, \omega')|$")
plt.xlabel("$\omega'$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar(orientation="horizontal")

ax = fig.add_subplot(1, 2, 2)
plt.imshow(np.abs(S), origin='lower', aspect="auto")
plt.title("$|S(\omega, \omega')|$")
plt.xlabel("$\omega'$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar(orientation="horizontal");

# %% [markdown]
# # Check that Green's function works
# Compare output signals obtained by simulating propagation and by multiplying the input (vector) by the Green's function matrix.

# %% [markdown]
# Define some arbitrary input shape

# %%
pulse = np.zeros(nFreqs)
pulse[nFreqs//4 : 3*nFreqs//4] = 1 - np.abs(np.arange(-nFreqs // 4, nFreqs // 4)) / (nFreqs // 4)
pulse = fftshift(pulse)
pulseFreq = fft(pulse)
fig = plt.figure()
plt.plot(fftshift(fiberS.tau), fftshift(pulse))
plt.title("Input Signal Temporal Field Profile")
plt.xlabel("time")
plt.ylabel("field")
plt.legend();

# %%
fig = plt.figure()
plt.plot(fftshift(fiberS.omega), fftshift(np.abs(pulseFreq)))
plt.title("Input Signal Spectral Field Profile")
plt.xlabel("angular frequency")
plt.ylabel("field")
plt.legend();

# %%
fiberS.runSignalSimulation(1j * pulse)
fig = plt.figure()
plt.plot(fftshift(fiberS.tau),  fftshift(np.abs(fiberS.signalTime[-1])), label="absolute value")
plt.plot(fftshift(fiberS.tau),  fftshift(np.real(fiberS.signalTime[-1])), label="real")
plt.plot(fftshift(fiberS.tau),  fftshift(np.imag(fiberS.signalTime[-1])), label="imaginary")

plt.title("Output Signal Temporal Field Profile from Simulation")
plt.xlabel("angular frequency")
plt.ylabel("field")
plt.legend();

# %%
output = C @ fftshift(1j * pulseFreq) + S @ fftshift((1j * pulseFreq).conj())
outputTime = fftshift(ifft(fftshift(output)))

# %%
fig = plt.figure()
plt.plot(fftshift(fiberS.tau), np.abs(outputTime),  label="absolute value")
plt.plot(fftshift(fiberS.tau), np.real(outputTime), label="real")
plt.plot(fftshift(fiberS.tau), np.imag(outputTime), label="imaginary")

plt.title("Output Signal Temporal Field Profile from Green's Function")
plt.xlabel("angular frequency")
plt.ylabel("field")
plt.legend();
