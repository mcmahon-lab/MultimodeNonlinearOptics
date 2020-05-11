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
# # Quantum signal propagation with $\chi$(3)

# %%
# cd ..

# %%
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft, ifft, fftshift, ifftshift

from NonlinearMedium import Chi3
print("using Python implementation")

# %%
# %matplotlib notebook
plt.rcParams['figure.figsize'] = [9, 6]

# %% [markdown]
# Consider no dispersion, and a real-valued pump, so
# $$
# \frac{\partial a}{\partial z} = i \gamma A^2 (2a + a^\dagger)
# $$
# and let $A^2 = \exp(-\tau^2)$
#
# Now let $a = b + i c$, such that
# $$
# \frac{\partial b}{\partial z} = - \gamma \exp(-\tau^2) c \\
# \frac{\partial c}{\partial z} = 3 \gamma \exp(-\tau^2) b
# $$
# In the case of $a(0) = 1$, the solution is:
# $$
# a = \cos \left(\sqrt{3} \gamma A^2 z \right) + i \sqrt{3} \sin \left(\sqrt{3} \gamma A^2 z \right)
# $$
# and in the case of $a(0) = i$,
# $$
# a = i \cos \left(\sqrt{3} \gamma A^2 z \right) - \frac{1}{\sqrt{3}} \sin \left(\sqrt{3} \gamma A^2 z \right)
# $$
#
# Or, in absolute terms
# $$
# a = \sqrt{2 - \cos \left(2 \sqrt{3} \gamma A^2 z \right) } \\
# a = \sqrt{ \frac{1}{3} \left( 2 + \cos \left(2 \sqrt{3} \gamma A^2 z \right) \right) }
# $$

# %%
zPres = 500

fiber = Chi3(relativeLength=1,
             nlLength=1,
             dispLength=np.inf,
             beta2=0,
             pulseType=0,
             tPrecision=512, zPrecision=zPres)

fiber.pumpTime[:] = np.exp(-0.5 * fiber.tau**2)

fiber.runSignalSimulation(np.ones(512))
curve1S = np.empty_like(fiber.signalTime[:, 0])
curve1S[:] = fiber.signalTime[:, 0]

fiber.runSignalSimulation(1j * np.ones(512))
curve2S = np.empty_like(curve1S)
curve2S[:] = fiber.signalTime[:, 0]

t = np.linspace(0, 1, zPres)
curve1 =  np.sqrt(2 - np.cos(2 * np.sqrt(3) * t))
curve2 = np.sqrt((2 + np.cos(2 * np.sqrt(3) * t)) / 3)


# %%
plt.figure()
plt.plot(np.abs(curve1S), label="simulated")
plt.plot(curve1, label="theory")
plt.title("Quantum Signal Field Profile, In Phase")
plt.xlabel("length")
plt.ylabel("field")
plt.legend();

# %%
plt.figure()
plt.plot(np.abs(curve2S), label="simulated")
plt.plot(curve2, label="theory")
plt.title("Quantum Signal Field Profile, Out of Phase")
plt.xlabel("length")
plt.ylabel("field")
plt.legend();
