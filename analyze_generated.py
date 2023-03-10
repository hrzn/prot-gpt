"""
A small script to compare the lengths of the generated sequences with the lengths of the real sequences.
"""

import sys
import matplotlib.pyplot as plt

PROT_FNAME = "data/prot_seqs.txt"  # needed for the vocab

with open(PROT_FNAME, "r") as f:
    lines = f.readlines()

with open("generated_prots/" + sys.argv[1], "r") as f:
    generated_lines = f.readlines()

plt.hist(
    [len(line) for line in lines if len(line) <= 1000],
    bins=50,
    density=True,
    alpha=0.5,
    label="real",
)
plt.hist(
    [len(line) for line in generated_lines if len(line) <= 1000],
    bins=50,
    density=True,
    alpha=0.5,
    label="generated",
)
plt.legend()
plt.xlabel("sequence length")
plt.ylabel("frequency")
plt.savefig("figures/" + sys.argv[1] + "_lengths.png", dpi=120)
plt.show()
