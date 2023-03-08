from collections import Counter
import numpy as np
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from nano_transformer import PLNanoTransformer
from pytorch_lightning import seed_everything

seed_everything(1337)

FOLDER = "pl_checkpoints"
FNAME = "nano_prot_gpt-epoch=00-step=00005000-val_loss=2.8516.ckpt"
PROT_FNAME = "data/prot_seqs.txt"  # needed for the vocab


######## build the vocab #########
# TODO: factor this out of train/generate files
with open(PROT_FNAME, "r") as f:
    lines = f.readlines()
pad = "!"  # padding character
flat_text = [c for line in lines for c in line]  # all proteins concatenated
chars = sorted(set(flat_text)) + [pad]
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}


def decode(i):
    return "".join([itos[ii] for ii in i])


# In order to generate, we compute the probability
# of each amino acid to appear in the first position
nr_times_first = Counter([line[0] for line in lines])
aa_to_proba_first = {aa: nr_times_first[aa] / len(lines) for aa in nr_times_first}


######## load the model (on CPU) #########

full_fname = os.path.join(FOLDER, FNAME)
device = torch.device("cpu")

# load this checkpoint with PyTorch Lightning
pl_model = PLNanoTransformer.load_from_checkpoint(full_fname, map_location=device)

pl_model.eval()


######## generate proteins #########


def generate_protein_string():
    # start_char_proba = {k: v / sum(counter.values()) for k, v in counter.items()}
    start_char = np.random.choice(
        list(aa_to_proba_first.keys()), p=list(aa_to_proba_first.values())
    )

    initial_context = torch.tensor([[stoi[start_char]]], dtype=torch.long)  # (1, 1)

    return decode(
        pl_model.model.generate_line(
            idx=initial_context,
            termination_token_idx=stoi["\n"],
            pad_token_idx=stoi["!"],
        ).tolist()[:-1]
    )  # remove the last \n


generated_seqs = []
for _ in tqdm(range(100)):
    generated_seqs.append(generate_protein_string())

with open("generated_proteins.txt", "w") as f:
    f.write("\n".join(generated_seqs))
