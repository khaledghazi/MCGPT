import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64 #how many independent samples to process at once
block_size = 256 #what is the maximum conetxt length for prediction
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
num_heads = 6
num_layers= 6
dropout = 0.2

torch.manual_seed(1337)
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

#define unique charachters
chars = sorted(list(set(text)))
vocab_size = len(chars)
#map chars to ints
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]#encode take str and output list of ints
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take list of int, output a str
#split dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data= data[n:]
#data loading
def get_batch(split):
  #generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,)) #produce a random int between batchsize and len data - blocksize
  x = torch.stack([data[i:i+block_size]for i in ix])
  y = torch.stack([data[i+1:i+block_size+1]for i in ix])
  x,y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
#class head

class Head(nn.Module):
  """A single self-attention head."""
  def __init__(self,head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size,bias=False)
    self.query = nn.Linear(n_embed, head_size,bias=False)
    self.value = nn.Linear(n_embed, head_size,bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # (B,T,C)
    q = self.query(x) # (B,T,C)
    #compute attention scores
    wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,T)
    #mask out future tokens
    wei =wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
    #normalize attention scores
    wei = F.softmax(wei, dim=-1) # (B,T,T)
    wei = self.dropout(wei) # (B,T,T)
    #apply attention scores to values
    v = self.value(x) # (B,T,C)
    out = wei @ v # (B,T,C)
    return out

#model

#multi head attention
class MultiHeadAttention(nn.Module):
  """multiple self-attention heads running in parallel."""

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    #concatenate the outputs of all heads
    out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,C)
    out = self.dropout(self.proj(out)) # (B,T,C)
    return out


class FeedForward(nn.Module):
  """A simple feed-forward network."""
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embed, 4 * n_embed),
        nn.ReLU(),
        nn.Linear(4 *n_embed, n_embed),
        nn.Dropout(dropout),
        
    )
  
  def forward(self, x):
    return self.net(x)

#class block
class Block(nn.Module):
  """A single Transformer block."""
  def __init__(self, n_embed, num_heads):
    super().__init__()
    head_size = n_embed // num_heads
    self.sa_heads = MultiHeadAttention(num_heads, head_size) # i.e e heads of 8-dimensional self-attention
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)
    
  def forward(self, x):
    x = x + self.sa_heads(self.ln1(x)) # (B,T,C) (Batch, Time, Channel)(id+pos+sa)
    x = x + self.ffwd(self.ln2(x)) # (B,T,C) (Batch, Time, Channel)(id+pos+sa+ffwd)
    return x

class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    #each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, num_heads =num_heads) for _ in range(num_layers)])
    self.ln_f = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)
  
  def forward(self, idx, targets=None):
    B, T = idx.shape # (Batch, Time)

    tok_emb = self.token_embedding_table(idx) # (B,T,C) (Batch, Time, Channel)(id)
    pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C) (Time, Channel)(pos)
    x= tok_emb + pos_emb # (B,T,C) (Batch, Time, Channel)(id+pos)
    x= self.blocks(x) # (B,T,C) (Batch, Time, Channel)(id+pos+sa)
    x= self.ln_f(x) # (B,T,C) (Batch, Time, Channel)(id+pos+sa)
    logits = self.lm_head(x) # (B,T,C) (Batch, Time, Channel)(id+pos+sa+lm)
    
    #C from tok_emb is not the same as C from logits (vocab_size)
    if targets is None:
      loss = None
    else:

      #but pytorch Cross_Entropy expect B,C,T
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T) # or -1 so 1 dim less 
      loss = F.cross_entropy(logits, targets) 
    return logits, loss

  def generate(self, idx, max_new_tokens):
    #idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      #crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:] # (B, T)
      #get preds
      logits, loss = self(idx_cond)
      #focus only on the las time step
      logits = logits[:, -1, :]# becomes (B, C) so T is omitted
      #apply softmat to rank probs
      probs = F.softmax(logits, dim=-1) #(B,C)
      #sample from a dist
      idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
      #append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim = 1) #(B, T+1)
    return idx

#training
model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    #everyonce in a while, evaluate the model on train and val set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'iter {iter} train loss {losses["train"]:.4f} val loss {losses["val"]:.4f}')
    
    #sampel a batch of data
    xb,yb = get_batch('train')

    #evalute the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


#generate text
context = torch.zeros((1,1), dtype=torch.long, device =device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))