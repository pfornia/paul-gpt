import torch
from statistics import mean

def get_encoder_decoder_size(text):
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




def text_to_tv_tensors(text, encode, device):
  data = torch.tensor(encode(text))

  val_cutoff = int(len(data)*0.9)
  train = data[:val_cutoff]
  validate = data[val_cutoff:]

  return train.to(device), validate.to(device)



def get_batch(
  data,
  block_size = 16,
  batch_size = 32,
):
  n = len(data)
  idxs = torch.randint(n - block_size, (batch_size,))

  xs = [data[i:i+block_size] for i in idxs]
  ys = [data[i+1:i+block_size+1] for i in idxs]

  return torch.stack(xs), torch.stack(ys)

#no_grad means don't calculate gradients
@torch.no_grad()
def print_loss_estimates(model, train_data, val_data, epoch=0, num_evals=100):
  model.eval() #eval mode, e.g., turns off drop out
  #list of X, Y pairs
  train_batches = [get_batch(train_data) for _ in range(num_evals)]
  #model returns logits, loss
  train_losses = [model(x[0], x[1])[1].item() for x in train_batches]

  #list of X, Y pairs
  val_batches = [get_batch(val_data) for _ in range(num_evals)]
  #model returns logits, loss
  val_losses = [model(x[0], x[1])[1].item() for x in val_batches]

  model.train() #goes back to train mode

  print(f"Epoch: {epoch}, Train Loss: {mean(train_losses):.4f}, Val Loss: {mean(val_losses):.4f}")

def training_run(
  model, 
  train_data, 
  val_data, 
  num_epochs=1000, 
  print_freq=None
):

  torch.manual_seed(42)

  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

  print(f"Training for {num_epochs} Epochs...")
  if not print_freq:
    print_freq = num_epochs//20

  for epoch in range(num_epochs):
    tx, ty = get_batch(train_data)

    _, loss = model(tx, ty)
    optimizer.zero_grad(set_to_none=True)
    if epoch%print_freq == 0:
      print_loss_estimates(model, train_data, val_data, epoch)
    loss.backward()
    optimizer.step()

def test_forward_pass(
  model,
  test_data
):
  tx, ty = get_batch(test_data)
  _, loss = model(tx, ty)
  print(f"Success! Loss = {loss}")

def test_gen_text(
  model,
  seed_raw,
  encode,
  decode
):
  seed = torch.tensor(encode(seed_raw)).view(1,-1)

  print(decode(model.generate(seed, 1000)[0,]))