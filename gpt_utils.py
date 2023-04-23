import torch

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




def text_to_tv_tensors(text, encode):
  data = torch.tensor(encode(text))
  print(data.shape, data.dtype)
  print(data[:100])

  val_cutoff = int(len(data)*0.9)
  train = data[:val_cutoff]
  validate = data[val_cutoff:]
  len(train), len(validate)

  return train, validate



def get_batch(
  data,
  BLOCK_SIZE = 16,
  BATCH_SIZE = 32,
):
  n = len(data)
  idxs = torch.randint(n - BLOCK_SIZE, (BATCH_SIZE,))

  xs = [data[i:i+BLOCK_SIZE] for i in idxs]
  ys = [data[i+1:i+BLOCK_SIZE+1] for i in idxs]

  return torch.stack(xs), torch.stack(ys)