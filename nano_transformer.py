"""
A nano GPT model, inspired from Karpathy's NanoGPT
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, head_size, block_size, n_embd, dropout):
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
    def __init__(self, num_heads, head_size, block_size, n_embd, dropout):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [Head(head_size, block_size, n_embd, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(
            n_embd, n_embd
        )  # linear layer after concat, as per transformer paper
        self.dropout = nn.Dropout(dropout)  # from Karpathy's video

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    # linear layer followed by nonlinearity
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                n_embd, 4 * n_embd
            ),  # in the paper the ffw layer is 4 times larger
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),  # from Karpathy's video
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.multi_head_attn = MultiHeadAttention(
            num_heads, head_size, block_size, n_embd, dropout
        )
        self.ffwd = FeedForward(n_embd, dropout)
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


class NanoTransformer(nn.Module):
    """
    A simple transformer made only of decoder (causal) blocks and no encoding block.
    """

    def __init__(self, vocab_size, block_size, n_embd, n_blocks, num_heads, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = torch.nn.Sequential(
            *[Block(n_embd, num_heads, block_size, dropout) for _ in range(n_blocks)],
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(
            n_embd, vocab_size
        )  # go from inner dim (concatenated) to logits
        self.block_size = block_size

    def forward(self, idx, targets=None):
        # idx: (B, T), targets: (B, T)
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C), with C = n_embd
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        x = self.blocks(x)  # (B, T, C)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        B, T, C = logits.shape

        loss = (
            F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
            if targets is not None
            else None
        )

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
