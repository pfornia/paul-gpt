import torch

import torch.nn as nn
from torch.nn import functional as F

from hyperparams import (
  BLOCK_SIZE,
  NUM_BLOCKS,
  N_EMB,
  N_HEAD,
  HEAD_SIZE,
  DROP_RATE,
)

torch.manual_seed(44)

class AttentionHead(nn.Module):

  def __init__(self, head_size):
    super().__init__()
    self.query = nn.Linear(N_EMB, head_size, bias = False)
    self.key = nn.Linear(N_EMB, head_size, bias = False)
    self.value = nn.Linear(N_EMB, head_size, bias = False)

    self.drop = nn.Dropout(DROP_RATE)

    #tril is just a constant, don't want it to "train"
    self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

  def forward(self, inputs):

    B, T, C = inputs.shape
    # print(B, T, C)
    # E.g., 4 x 8 x 63 = examples/batch x chars per example (time) x vocab_size

    q = self.query(inputs) #B x T x head_size
    k = self.key(inputs) #B x T x head_size
    v = self.value(inputs) # B x T x head_size

    wei = q @ torch.transpose(k, 1, 2) * C**-0.5 # B x T X T

    wei = wei.masked_fill(self.tril == 0, float('-inf'))

    wei = torch.softmax(wei, dim=-1) # BxTxT

    wei = self.drop(wei)

    out = wei @ v # B x T x head_size 

    return out


class AttentionMultiHead(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])

    self.mh_seq = nn.Sequential(
        nn.Linear(N_EMB, N_EMB), #Fuzzy on why I need this... Should always do a linear before an addition?
        nn.Dropout(DROP_RATE)
    )

  def forward(self, inputs):
    heads_concat = torch.cat([head(inputs) for head in self.heads], dim=-1)

    return self.mh_seq(heads_concat)

class AttentionDecodeBlock(nn.Module):
  def __init__(self):
    super().__init__()

    self.block_seq1 = nn.Sequential(
        nn.LayerNorm(N_EMB), #Karpathy vid suggests deviating from AIAYN: normalize before multi-head and feed forward, not after.
        AttentionMultiHead(N_HEAD, HEAD_SIZE)
    )

    self.block_seq2 = nn.Sequential(
        nn.LayerNorm(N_EMB),
        nn.Linear(N_EMB, 4*N_EMB), # section 3.3 of AIAYN, bigger "inner" dimension.
        nn.ReLU(),
        nn.Linear(4*N_EMB, N_EMB),
        nn.Dropout(DROP_RATE)
    )

  def forward(self, inputs):
    
    mhead = self.block_seq1(inputs)

    mhead_add = mhead + inputs

    lin_out = self.block_seq2(mhead_add)

    lin_add = lin_out + mhead_add #skip connection

    return lin_add


class AttentionModule(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.vocab_size = vocab_size
    self.tok_embedding = nn.Embedding(vocab_size, N_EMB)
    self.pos_embedding = nn.Embedding(BLOCK_SIZE, N_EMB)

    self.seq_of_enc_blocks = nn.Sequential(
        *[AttentionDecodeBlock() for _ in range(NUM_BLOCKS)],
    )

    self.norm_lin = nn.Sequential(
        nn.LayerNorm(N_EMB),
        nn.Linear(N_EMB, vocab_size),
    )
    # self.layer_norm1 = 
    # self.lin1 = 

  def forward(self, inputs, targets=None):

    B, T = inputs.shape
    # E.g., 4 x 8 x 63 = examples/batch x chars per example (time) x vocab_size

    tok_emb = self.tok_embedding(inputs)
    pos_emb = self.pos_embedding(torch.arange(BLOCK_SIZE))

    inputs_emb = tok_emb + pos_emb

    block = self.seq_of_enc_blocks(inputs_emb)
    logits = self.norm_lin(block)

    # It may seem odd that these targets are indices, where logits are scores
    # Basically, pytorch can handle EITHER indices as targets, or the actual OHE/probability ground truths. 
    #   could try target smoothing by OHE to 0.01 and 0.99 instead of 0/1?
    
    if targets is None:
      loss = None
    else:
      # loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
      # print("loss stuff")
      # print(logits.shape)
      # print(targets.shape)
      loss = F.cross_entropy(logits.view(B*T, self.vocab_size), targets.view(B*T))
      # loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

    return logits, loss

  def generate(self, inputs, max_new_tokens):
    for _ in range(max_new_tokens):
      # print(inputs.shape)
      logits, _ = self(inputs[:,(-1*BLOCK_SIZE):]) #B, T, C
      # print(logits.shape)
      logits_last = logits[:,-1] #end of string only
      probs_last = F.softmax(logits_last, dim=1)
      next_char = torch.multinomial(probs_last, num_samples=1)

      inputs = torch.cat((inputs, next_char), dim=1) #B, T+1

    return inputs

