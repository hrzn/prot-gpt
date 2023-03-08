import matplotlib.pyplot as plt

PROT_FNAME = "data/prot_seqs.txt"  # needed for the vocab

# TODO: factor this out of train/generate files
with open(PROT_FNAME, "r") as f:
    lines = f.readlines()

with open("generated_proteins.txt", "r") as f:
    generated_lines = f.readlines()

plt.hist([len(line) for line in lines], bins=20, density=True, alpha=0.5, label="real")
plt.hist(
    [len(line) for line in generated_lines],
    bins=20,
    density=True,
    alpha=0.5,
    label="generated",
)
plt.legend()
plt.show()
