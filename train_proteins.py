"""
Training a simple NanoGPT model on protein sequences

Adapted from Karpathy's NanoGPT (https://github.com/karpathy/nanoGPT) with some adaptations:
* Vocabulary is amino acids
* Make it trainable on multiple isolated sequences of variable lengths, using padding and masking
* Use PyTorch Lightning for training loop

The dataset comes from
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

import torch
import random
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import Sampler
from typing import Tuple
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

from nano_transformer import NanoTransformer, PLNanoTransformer


seed_everything(1337)

PROT_FNAME = "data/prot_seqs.txt"

# hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 if DEVICE == "cpu" else 64
BLOCK_SIZE = 8 if DEVICE == "cpu" else 256  # context size
N_EMBD = 16 if DEVICE == "cpu" else 384  # called d_model in paper
N_BLOCKS = 2 if DEVICE == "cpu" else 6  # number N of transformer blocks
NUM_HEADS = 2 if DEVICE == "cpu" else 6  # nr attention heads
DROPOUT = 0.2
MAX_ITERS = 5000
EVAL_INTERVAL = 500
# LEARNING_RATE = 3e-4
LEARNING_RATE = 1e-3 if DEVICE == "cpu" else 3e-4
EVAL_ITERS = 200
PRECISION = 32
CHECKPOINT_EVERY_STEP = 500
print(f"device: {DEVICE}")


""" Read file
"""

with open(PROT_FNAME, "r") as f:
    lines = f.readlines()


""" Define amino acid vocabulary encoding
"""

pad = "!"  # padding character
flat_text = [c for line in lines for c in line]  # all proteins concatenated
chars = sorted(set(flat_text)) + [pad]
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

counter = Counter(flat_text)
# plt.bar(counter.keys(), counter.values(), log=True)


def encode(s):
    return [stoi[c] for c in s]


def decode(i):
    return "".join([itos[ii] for ii in i])


def encode_pad(s, block_size):
    encoding = encode(s)
    return encoding + [stoi[pad]] * max(0, block_size + 1 - len(encoding))


""" Train / val split
"""

# There is one line per protein.
# We shuffle for train/val split (done in place).
random.shuffle(lines)
n = int(0.9 * len(lines))
train_lines = lines[:n]
val_lines = lines[n:]


""" Define dataset.
    We need to define a custome PyTorch `Dataset` which disambiguates 
    between line indices and subsequence start indices.
"""


class LineDataset(Dataset):
    """A dataset where sampling is uniform over all lines.
    All lines are sampled with the same frequency.
    """

    def __init__(self, lines):
        self.lines = lines
        nr_samples_per_line = [len(line) - BLOCK_SIZE for line in self.lines]
        self.num_samples = sum(nr_samples_per_line)
        # self.max_line_length = max(len(line) - BLOCK_SIZE for line in lines)

    def __len__(self):
        # return len(self.lines) * self.max_line_length
        return self.num_samples

    def __getitem__(self, idx):
        # select a line uniformly at random and then a subsequence uniformly at random
        # use idx to setup the torch random seed
        generator = torch.Generator()
        generator.manual_seed(idx)
        line_idx = torch.randint(len(self.lines), size=(1,), generator=generator).item()
        line = self.lines[line_idx]
        line_encoded = encode_pad(line, BLOCK_SIZE)
        start_idx = torch.randint(
            len(line_encoded) - BLOCK_SIZE, size=(1,), generator=generator
        ).item()
        end_idx = start_idx + BLOCK_SIZE
        x = torch.tensor(line_encoded[start_idx:end_idx], dtype=torch.long)
        y = torch.tensor(line_encoded[start_idx + 1 : end_idx + 1], dtype=torch.long)
        length = torch.tensor(min(len(line), BLOCK_SIZE), dtype=torch.long)
        return x, y, length


class LengthAwareLineDataset(Dataset):
    """A dataset where sampling is uniform over all possible subsequences.
    As a result, sequences are (much) more likely to come from longer lines.
    """

    def __init__(self, lines):
        self.lines = sorted(lines, key=len, reverse=True)
        nr_samples_per_line = [len(line) - BLOCK_SIZE for line in self.lines]
        self.num_samples = sum(nr_samples_per_line)
        self.line_idx_to_cumulative_nr_samples = np.array(
            [0] + np.cumsum(nr_samples_per_line).tolist()
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # infer line idx from idx, using binary search over line_idx_to_cumulative_nr_samples
        # np.searchsorted() returns the index of the first element in the array that is greater than the value
        line_idx = (
            np.searchsorted(self.line_idx_to_cumulative_nr_samples, idx, side="right")
            - 1
        )
        start_idx = idx - self.line_idx_to_cumulative_nr_samples[line_idx]

        line = self.lines[line_idx]
        line_encoded = encode_pad(line, BLOCK_SIZE)

        x = torch.tensor(
            line_encoded[start_idx : start_idx + BLOCK_SIZE], dtype=torch.long
        )
        y = torch.tensor(
            line_encoded[start_idx + 1 : start_idx + BLOCK_SIZE + 1], dtype=torch.long
        )
        length = torch.tensor(min(len(line), BLOCK_SIZE), dtype=torch.long)
        return x, y, length


inner_model = NanoTransformer(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_blocks=N_BLOCKS,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
)

nano_prot_gpt = PLNanoTransformer(inner_model, LEARNING_RATE)

train_dataset = LineDataset(train_lines)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = LineDataset(val_lines)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# configure tensorboard logging
logger = TensorBoardLogger("tensorboard_logs", name="nano_prot_gpt")

# log hyperparameters
logger.log_hyperparams(
    {
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD,
        "n_blocks": N_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout": DROPOUT,
        "max_iters": MAX_ITERS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "precision": PRECISION,
    }
)

# configure model checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath="pl_checkpoints",
    save_top_k=2,
    monitor="val_loss",
    mode="min",
    every_n_train_steps=CHECKPOINT_EVERY_STEP,
    filename="nano_prot_gpt-{epoch:02d}-{step:08d}-{val_loss:.4f}",
)

trainer = pl.Trainer(
    max_steps=MAX_ITERS,
    val_check_interval=EVAL_INTERVAL,
    limit_val_batches=EVAL_ITERS,
    callbacks=[TQDMProgressBar(refresh_rate=40), checkpoint_callback],
    precision=PRECISION,
    logger=logger,
    accelerator=DEVICE,
)

trainer.fit(
    model=nano_prot_gpt, train_dataloaders=train_loader, val_dataloaders=val_loader
)

print("Best model checkpoint:", checkpoint_callback.best_model_path)
