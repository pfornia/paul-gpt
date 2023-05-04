import torch
from statistics import mean
import random

from .hyperparams import (
  BLOCK_SIZE,
  BATCH_SIZE,
  NUM_EPOCHS,
  LEARNING_RATE,
)

from transformers import AutoTokenizer

def wiki_text_clean(article_text):
  return article_text.split("\nReferences")[0]

def get_encoder_decoder_size_char(text):
  """
  Use character-level encoder
  """
  vocab = sorted(list(set(text)))
  # print(''.join(vocab))
  vocab_size = len(vocab)
  # vocab_size
  str2code = {c: i for i, c in enumerate(vocab)}
  code2str = {i: c for i, c in enumerate(vocab)}

  encode = lambda text: [str2code[c] for c in text]
  def decode(code_list): 
    if type(code_list) == torch.Tensor:
      return decode(code_list.tolist())
    if type(code_list) == int:
      return code2str[code_list]
    return ''.join([code2str[i] for i in code_list]) 

  return encode, decode, vocab_size


def get_encoder_decoder_size_gpt2(text):
  """
  use fast gpt2 tokenizer from hugging face 
  Tokenizes pieces of words (aka subwords), with 50k vocab size

  Read this for more on :)> tokenizers https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt
  """
  tokenizer = AutoTokenizer.from_pretrained("gpt2")

  encode = lambda text: tokenizer(text)['input_ids']
  
  return encode, tokenizer.decode, tokenizer.vocab_size



def get_encoder_decoder_size(
    text,
    option = "char"
):
  if option == "char":
    return get_encoder_decoder_size_char(text)
  elif option == "gpt2":
    return get_encoder_decoder_size_gpt2(text)
  else:
    raise ValueError(f"Option '{option}' not supported for encoder/decoder. Try 'char' for character level, or 'gpt2' for word part tokenizer.")


def text_to_tv_tensors(text, encode, n_chunks=10):
  val_cutoff = int(len(text)*0.9)

  train_text = text[:val_cutoff]
  val_text = text[val_cutoff:]

  chunk_size = val_cutoff//n_chunks

  train_chunks = []
  for i in range(n_chunks):
    chunk_text = train_text[i*chunk_size:(i+1)*chunk_size]
    train_chunks.append(
      torch.tensor(encode(chunk_text))
    )

  return (
    train_chunks, 
    torch.tensor(encode(val_text))
  )



def get_batch(
  data,
  device = torch.device("cpu"),
  block_size = BLOCK_SIZE,
  batch_size = BATCH_SIZE,
):
  n = len(data)
  idxs = torch.randint(n - block_size, (batch_size,))

  xs = [data[i:i+block_size] for i in idxs]
  ys = [data[i+1:i+block_size+1] for i in idxs]

  return torch.stack(xs).to(device), torch.stack(ys).to(device)

#no_grad means don't calculate gradients
@torch.no_grad()
def print_loss_estimates(model, train_data, val_data, device, epoch=0, num_evals=100):
  model.eval() #eval mode, e.g., turns off drop out
  #list of X, Y pairs
  train_batches = [get_batch(train_data, device=device) for _ in range(num_evals)]
  #model returns logits, loss
  train_losses = [model(x[0], x[1])[1].item() for x in train_batches]

  #list of X, Y pairs
  val_batches = [get_batch(val_data, device=device) for _ in range(num_evals)]
  #model returns logits, loss
  val_losses = [model(x[0], x[1])[1].item() for x in val_batches]

  model.train() #goes back to train mode

  print(f"Epoch: {epoch}, Train Loss: {mean(train_losses):.4f}, Val Loss: {mean(val_losses):.4f}")

def training_run(
  model, 
  train_chunks, 
  val_data,
  device,
  num_epochs=NUM_EPOCHS, 
  print_freq=None,
  learning_rate=LEARNING_RATE
):

  torch.manual_seed(42)

  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

  print(f"Training for {num_epochs} Epochs...")
  if not print_freq:
    print_freq = max(1, num_epochs//20)

  for epoch in range(num_epochs):
    tx, ty = get_batch(random.choice(train_chunks), device=device)

    _, loss = model(tx, ty)
    optimizer.zero_grad(set_to_none=True)
    if epoch%print_freq == 0:
      print_loss_estimates(model, random.choice(train_chunks), val_data, device, epoch, num_evals=min(print_freq, 100))
    loss.backward()
    optimizer.step()

def test_forward_pass(
  model,
  test_data,
  device,
  batch_size=BATCH_SIZE,
):
  tx, ty = get_batch(test_data, device=device, batch_size=batch_size)
  _, loss = model(tx, ty)
  print(f"Success! Loss = {loss}")

def test_gen_text(
  model,
  seed_raw,
  encode,
  decode,
  device, 
  block_size=BLOCK_SIZE,
  n_out_tokens = 500,
):
  _ = model.eval()

  seed_encode = encode(seed_raw)

  if len(seed_encode) < block_size:
    seed_encode = encode(" ")*(block_size - len(seed_encode)) + seed_encode
  elif len(seed_encode) > block_size:
    seed_encode = seed_encode[-block_size:]

  seed = torch.tensor(seed_encode).view(1,-1).to(device)

  print(decode(model.generate(seed, n_out_tokens)[0,]))

  _ = model.train()