from sys import exec_prefix
import numpy as np
import matplotlib.pyplot as plt


off_book=False

try:
  get_ipython()
  off_book=False
except NameError:
  off_book=True

print("Is out of notebook ", off_book)

#|%%--%%| <IF8jHi7Hqu|KvN7PAkw05>

ABC_LPMT = (0.0261, 0.0082, 0.0123)
ABC_SPMT = (0.1536, 0.0082, 0.0677)

#|%%--%%| <KvN7PAkw05|StTnS4Ih6s>

def abc_fn(x, A: float, B: float, C: float):
  return np.sqrt((A / np.sqrt(x)) ** 2 + B**2 + (C/x)**2)

#|%%--%%| <StTnS4Ih6s|wgzZC3UtFD>

e_range = np.arange(0.8, 9, 0.02)

if off_book:
  fig = plt.figure("REL_ABC", clear=True, figsize=(8,6), dpi=300)
else:
  fig = plt.figure("REL_ABC", clear=True)

plt.plot(e_range, abc_fn(e_range, *ABC_LPMT), color='tab:blue', label=f"LPMT {ABC_LPMT}")
plt.plot(e_range, abc_fn(e_range, *ABC_SPMT), color='tab:orange', label=f"SPMT {ABC_SPMT}")
plt.axhline(0.03, linestyle='--', color='black', label="3% resolution",)
plt.grid(True)
plt.ylim(0, 0.20)
plt.xlim(0.8, 9)
plt.xticks(np.arange(1, 9, 0.5))
plt.ylabel("$\\sigma E_{vis} / E_{vis}$")
plt.xlabel("$E_{vis}$ [MeV]")
plt.title("ABC relative resolution of the two systems of JUNO")
plt.legend()
if off_book:
  fig.savefig("plots/relative_resolution.png")


if off_book:
  fig = plt.figure("ABS_ABC", clear=True, figsize=(8,6), dpi=300)
else:
  fig = plt.figure("ABS_ABC", clear=True)

plt.plot(e_range, e_range * abc_fn(e_range, *ABC_LPMT), color='tab:blue', label=f"LPMT {ABC_LPMT}")
plt.plot(e_range, e_range * abc_fn(e_range, *ABC_SPMT), color='tab:orange', label=f"SPMT {ABC_SPMT}")
plt.grid(True)
plt.ylim(0, 0.5)
plt.xlim(0.8, 9)
plt.xticks(np.arange(1, 9, 0.5))
plt.ylabel("$\\sigma E_{vis}$ [Mev]")
plt.xlabel("$E_{vis}$ [MeV]")
plt.title("ABC absolute resolution of the two systems of JUNO")
plt.legend()
if off_book:
  fig.savefig("plots/absolute_resolution.png")

#if off_book:
#  fig = plt.figure("SPMT_ABC", clear=True, figsize=(8,6), dpi=300)
#else:
#  fig = plt.figure("SPMT_ABC", clear=True)
