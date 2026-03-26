import numpy as np
import matplotlib.pyplot as plt
import ast

# ---- Load your file ----
fname = "/Users/xueshicong/Documents/research/Research-main/Vg_sims_n_dimension_effects_correct_L100_rep100"

# Read parameters from header
with open(fname, 'r') as f:
    header = f.readline()
    params = eval(header[2:].strip(), {"np": np})

# Load Vg values
Vg_sims = np.loadtxt(fname)

# If it's a single row, make sure it's 1D
Vg_values = Vg_sims.flatten()

# ---- Make violin plot ----
plt.figure(figsize=(5,6))

plt.violinplot([Vg_values],
               showmeans=True,
               showmedians=True)

# Labels
plt.xticks([1], ['Your simulation'])
plt.ylabel(r'$V_g$')

plt.title('Distribution of genetic variance across replicates')

plt.tight_layout()
plt.show()