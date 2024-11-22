import torch
import torch.nn as nn
import torch.optim as optim
import math, time
import numpy as np
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset

# Smaller model parameters for mobile
batch_size = 10
eval_batch_size = 5
best_model_path = 'best_model.pth'
early_stopping_patience = 3  # shorter patience for quicker iterations

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def data_process(raw_text_iter):
  data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def train(model, train_data, batch_size, bptt, optimizer, criterion, clip):
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, len(train_data) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, len(vocab)), targets)
        loss.backward()
        total_loss += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if batch % 200 == 0 and batch > 0:
            cur_loss = total_loss / 200
            elapsed = time.time() - start_time
            print(f'| batch {batch:3d} | loss {cur_loss:5.2f} | time {elapsed:5.2f}s')
            total_loss = 0
            start_time = time.time()

def evaluate(model, val_data, batch_size, bptt, criterion):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, len(val_data) - 1, bptt):
            data, targets = get_batch(val_data, i, bptt)
            output = model(data)
            total_loss += criterion(output.view(-1, len(vocab)), targets).item()
    return total_loss / (len(val_data) // bptt)

# Hyperparameters
ntokens = len(vocab)
emsize = 200
nhid = 200
nlayers = 2
nhead = 2
dropout = 0.2
bptt = 35
lr = 5.0
epochs = 3
clip = 0.25

# Initialize the model
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Early stopping setup
early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=best_model_path)

# Training loop
for epoch in range(epochs):
    epoch_start_time = time.time()
    train(model, train_data, batch_size, bptt, optimizer, criterion, clip)
    val_loss = evaluate(model, val_data, batch_size, bptt, criterion)
    print(f'| epoch {epoch+1:3d} | time {time.time() - epoch_start_time:5.2f}s | val_loss {val_loss:5.2f}')
    
    # Early stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Load the best model
model.load_state_dict(torch.load(best_model_path))

# Test the model
test_loss = evaluate(model, test_data, batch_size, bptt, criterion)
print(f'Test loss: {test_loss:.2f}')
