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

    def forward(self, x, lengths=None):
        # inputs:
        # x: (B, T, d_model) the tokens embeddings
        # lengths: (B, ); the original lengths of the sequences in the batch
        #                 (before padding). Needed for masking.

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

        # mask padding tokens to avoid communication between them and other tokens
        # we still want to make it possible for tokens to be self-attending
        # in order to avoid normalization issues with softmax.
        if lengths is not None:
            # (B, T)
            mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]

            # (B, 1, T)
            mask = mask[:, None, :]

            # (B, T, T)
            mask = mask & mask.transpose(-2, -1)

            # (B, T, T), enable diagonal entries for tokens to self-attend
            # (even padded ones, otherwise softmax normalization will yield NaNs)
            mask = mask | torch.eye(T, device=x.device, dtype=torch.bool)[None, :, :]

            weights = weights.masked_fill(~mask, float("-inf"))

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

    def forward(self, x, lengths=None):
        out = torch.cat([head(x, lengths) for head in self.heads], dim=-1)
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

    def forward(self, x, lengths=None):
        # the original transformer implements layer norm after the transformation
        # but here we implement the "pre-norm" version, now slightly more common,
        # where the layer norm is applied before the transformation

        # masked multi-head attn
        x = x + self.multi_head_attn(self.ln1(x), lengths)

        # feed forward
        x = x + self.ffwd(self.ln2(x))
        return x


class SequentialWithLengths(nn.Sequential):
    # our own implementation of Sequential supporting two inputs
    def forward(self, x, lengths=None):
        for module in self:
            x = module(x, lengths)
        return x


class NanoTransformer(nn.Module):
    """
    A simple transformer made only of decoder (causal) blocks and no encoding block.
    """

    def __init__(self, vocab_size, block_size, n_embd, n_blocks, num_heads, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = SequentialWithLengths(
            *[Block(n_embd, num_heads, block_size, dropout) for _ in range(n_blocks)]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(
            n_embd, vocab_size
        )  # go from inner dim (concatenated) to logits
        self.block_size = block_size

    def forward(self, idx, targets=None, lengths=None):
        # idx: (B, T), lengths: (B,), targets: (B, T)
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C), with C = n_embd
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        x = self.blocks(x, lengths)  # (B, T, C)
        x = self.ln(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        B, T, C = logits.shape

        loss = (
            F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
            if targets is not None
            else None
        )

        return logits, loss

    def generate_line(self, idx, termination_token_idx):
        while idx[0][-1].item() != termination_token_idx:
            logits, _ = self(idx[:, -self.block_size :])
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
