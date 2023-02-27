import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32 if device == "cpu" else 64
block_size = 8 if device == "cpu" else 256  # context size
n_embd = 32 if device == "cpu" else 384  # called d_model in paper
n_blocks = 3 if device == "cpu" else 6  # number N of transformer blocks
# head_size = 32  # called d_k = d_v
num_heads = 4 if device == "cpu" else 6  # nr attention heads
dropout = 0.2
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3 if device == "cpu" else 3e-4

eval_iters = 200
print(f"device: {device}")
# -------------

torch.manual_seed(1337)

# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("data/tiny_shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]
def decode(i):
    return "".join([itos[ii] for ii in i])

# train / val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)  # from Karpathy's video
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x: (B, T, n_embd)
        # output: (B, T, d_model)
        B, T, _ = x.shape

        keys = self.key(x)  # (B, T, d_model)
        queries = self.query(x)  # (B, T, d_model)
        values = self.value(x)  # (B, T, C)
        weights = queries @ keys.transpose(-2, -1)  # (B, T, T)

        # make weights causal
        # This means we are implementing a decoder block here
        # (in encoder blocks all tokens can talk to each other)
        # Note: we take last T entries in tril, in case T < block_size
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = weights * self.head_size**-0.5

        # apply softmax
        weights = F.softmax(weights, dim=-1)

        weights = self.dropout(weights)  # from Karpathy's video

        # multiplty with values
        return weights @ values
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # linear layer after concat, as per transformer paper
        self.dropout = nn.Dropout(dropout)  # from Karpathy's video
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    # linear layer followed by nonlinearity
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # in the paper the ffw layer is 4 times larger
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)  # from Karpathy's video
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.multi_head_attn = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # the original transformer implements layer norm after the transformation
        # but here we implement the "pre-norm" version, now slightly more common,
        # where the layer norm is applied before the transformation

        # masked multi-head attn
        x = x + self.multi_head_attn(self.ln1(x))

        # feed forward
        x = x + self.ffwd(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = torch.nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_blocks)], nn.LayerNorm(n_embd))
        self.lm_head = nn.Linear(n_embd, vocab_size)  # go from inner dim (concatenated) to logits

    def forward(self, idx, targets=None):
        # idx: (B, T), targets: (B, T)
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C), with C = n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        x = self.blocks(x)  # (B, T, C)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        B, T, C = logits.shape

        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) if targets is not None else None

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
    

model = SimpleTransformer()
model = model.to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

tic = time.time()
for iter in range(max_iters):
    # evaluate loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=None)
    loss.backward()
    optimizer.step()
print(f"training duration: {(time.time() - tic):.2f} s.")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))