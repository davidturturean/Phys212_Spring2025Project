# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 23:21:10 2025

@author: Daria Teodora Harabor
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Load chains and trim burn-in
# --------------------
chains = np.load("chains_lambdaCDM.npy")
n_chains, n_steps, n_params = chains.shape

burn_in = 2000
trimmed_chains = chains[:, burn_in:, :]

# --------------------
# Plot histograms of posteriors
# --------------------
labels = ["$A_s$", "$n_s$", "$\\tau$"]
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for i in range(n_params):
    for j in range(n_chains):
        axs[i].hist(trimmed_chains[j, :, i], bins=40, alpha=0.6, label=f"Chain {j+1}")
    axs[i].set_xlabel(labels[i])
    axs[i].set_ylabel("Frequency")
axs[0].legend()
plt.tight_layout()
plt.show()

# --------------------
# Gelman-Rubin (R-hat) diagnostic
# --------------------
def gelman_rubin(chains):
    m, n, d = chains.shape  # m chains, n samples, d parameters
    R_hat = []
    for i in range(d):
        chain_means = np.mean(chains[:, :, i], axis=1)
        chain_vars = np.var(chains[:, :, i], axis=1, ddof=1)
        B = n * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)
        V_hat = (1 - 1/n) * W + B / n
        R = np.sqrt(V_hat / W)
        R_hat.append(R)
    return np.array(R_hat)

rhat = gelman_rubin(trimmed_chains)

print("\nGelman-Rubin R-hat values:")
for i, name in enumerate(labels):
    print(f"{name}: {rhat[i]:.3f}")
