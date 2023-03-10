"""
Preprocess the PDB sequence data.
original source https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz
"""

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

SEQ_FILE = "data/pdb_seqres.txt"

with open(SEQ_FILE) as f:
    lines = f.readlines()

# We keep `mol:protein` types, and only one of each name (the first one)
names = set()

name_to_seq = dict()

for l in tqdm(lines):
    if l.startswith(">"):
        keep = False
        is_prot = "mol:protein" in l
        if is_prot:
            name = " ".join(l.split()[3:])
            if name not in names:
                names.add(name)
                keep = True
    elif keep:
        name_to_seq[name] = l.strip()


# Write file:
with open("data/prot_seqs.txt", "w") as f:
    f.write("\n".join(name_to_seq.values()))

print("number sequences:", len(name_to_seq))
print("number tokens:", len(set("".join(name_to_seq.values()))))

lengths = [len(seq) for seq in name_to_seq.values()]
plt.hist(lengths, bins=100, log=True)
plt.title(
    "distribution of protein sequence lengths ($\mu$={:.2f})".format(np.mean(lengths))
)
plt.xlabel("sequence length")
plt.ylabel("number of sequences")
plt.show()
