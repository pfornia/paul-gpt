import torch

import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(44)

# See Karpathy 1:40:00 for some suggested hyperparams.

BLOCK_SIZE = 16
HEAD_SIZE = 16
N_EMB = 32
DROP_RATE = 0.2

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
  def __init__(self, num_heads):
    super().__init__()
    head_size = N_EMB//num_heads
    self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
    self.lin1 = nn.Linear(N_EMB, N_EMB) #Fuzzy on why I need this... Should always do a linear before an addition?
    self.drop = nn.Dropout(DROP_RATE)

  def forward(self, inputs):
    heads_concat = torch.cat([head(inputs) for head in self.heads], dim=-1)
    heads_lin = self.lin1(heads_concat)
    return self.drop(heads_lin)

class AttentionDecodeBlock(nn.Module):
  def __init__(self):
    super().__init__()
    
    
    self.layer_norm1 = nn.LayerNorm(N_EMB)
    self.multi_head = AttentionMultiHead(4)
    

    self.layer_norm2 = nn.LayerNorm(N_EMB)
    # self.lin1 = nn.Linear(32, 16)
    self.lin1 = nn.Linear(N_EMB, 4*N_EMB) # section 3.3 of AIAYN, bigger "inner" dimension.
    self.relu = nn.ReLU()
    self.lin2 = nn.Linear(4*N_EMB, N_EMB)
    self.drop = nn.Dropout(DROP_RATE)

  def forward(self, inputs):
    
    norm_inputs = self.layer_norm1(inputs) #Karpathy vid suggests deviating from AIAYN: normalize before multi-head and feed forward, not after.
    mhead = self.multi_head(inputs)
    mhead_add = mhead + inputs

    mhead_norm = self.layer_norm2(mhead_add)    
    lin1_out = self.lin1(mhead_norm)
    relu_out = self.relu(lin1_out)
    lin2_out = self.lin2(relu_out) #Fuzzy on why I need this... Should always do a linear before an addition?
    lin2_drop = self.drop(lin2_out)
    lin_add = lin2_out + mhead_add #skip
    lin_norm = lin_add
    # lin_norm = self.layer_norm(lin_add)


    return lin_norm


class AttentionModule(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.vocab_size = vocab_size
    self.tok_embedding = nn.Embedding(vocab_size, N_EMB)
    self.pos_embedding = nn.Embedding(BLOCK_SIZE, N_EMB)
    self.block1 = AttentionDecodeBlock()
    self.block2 = AttentionDecodeBlock()
    self.block3 = AttentionDecodeBlock()

    self.layer_norm1 = nn.LayerNorm(N_EMB)
    self.lin1 = nn.Linear(N_EMB, vocab_size)

  def forward(self, inputs, targets=None):

    B, T = inputs.shape
    # E.g., 4 x 8 x 63 = examples/batch x chars per example (time) x vocab_size

    tok_emb = self.tok_embedding(inputs)
    pos_emb = self.pos_embedding(torch.arange(BLOCK_SIZE))

    inputs_emb = tok_emb + pos_emb

    # print(embedded_input.shape)

    # fc1 = self.lin1(embedded_input)
    # 4 x 8 x 16

    # fc1_flat = fc1.view(B, -1) #4, 8*16= 128
    # print(fc1_flat.shape)
    # fc2 = self.lin2(fc1_flat) # 4, 65

    b1 = self.block1(inputs_emb)
    b2 = self.block2(b1)
    b3 = self.block3(b2)

    b3_norm = self.layer_norm1(b3)
    logits = self.lin1(b3_norm)

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

