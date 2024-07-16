import torch
from tokenizer.BPE import MinBPE
from w2v_cbow.embed_train import Cbow
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import NLLLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import datetime

# Variable to hold all the text data
all_text: str

# Importing text file for data
with open('./data/taylor.txt', 'r') as f:
    all_text = f.read()

# Paths to the vocabulary and merge files
vocab_path = r'data/2024-07-13-T-08-55_vocab.pkl'
merges_path = r'data/2024-07-13-T-08-55_merges.pkl'

# Initializing BPE tokenizer
obj = MinBPE(vocab_path, merges_path)

# Encoding the text using BPE tokenizer
text_encoded = obj.encode(all_text)

# Configuration for the CBOW model
config = {
    "vocab_size": len(obj.vocab),
    "d_model": 128,
    "inter_dim": 4,
    "window_size": 2
}

# Preparing training data (context-target pairs)
X, y_full = [], []
for i in range(config['window_size'], len(text_encoded) - config['window_size']):
    # Context words (window around the target word)
    x = torch.tensor(text_encoded[i - config['window_size']:i] + text_encoded[i + 1:i + config['window_size'] + 1])
    # Target word
    y = torch.tensor(text_encoded[i])
    X.append(x)
    y_full.append(y)

# Converting lists to tensors
X = torch.stack(X)
Y = torch.stack(y_full)

# Creating a dataset and dataloader
my_dataset = TensorDataset(X, Y)
my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)

# Setting device to GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initializing the CBOW model and moving it to the appropriate device
model = Cbow(config['vocab_size'], config['d_model'])
model = model.to(device)

# Initializing the optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
criterion = NLLLoss()

# Number of training epochs
Total_epochs = 10
tot_loss = []

# Training loop
model.train()
for epoch in range(Total_epochs):
    loss_epoch = 0
    progress = tqdm(enumerate(my_dataloader), total=len(my_dataloader), colour='green')
    for i, (x_data, y_data) in progress:
        optimizer.zero_grad()
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        progress.set_postfix({'loss': loss.item()})
        progress.set_description(f'Epoch: {epoch + 1}/{Total_epochs}')
    tot_loss.append(loss_epoch / len(my_dataloader))

# Moving the model back to CPU for inference
model = model.cpu()

# Encoding a sample text
text = " man girl boy women"
encoded_text = obj.encode(text)
print(encoded_text)

# saving the model
with open(f'./data/{datetime.datetime.now().strftime("%Y-%m-%d-T-%H-%M")}_embed_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Inference mode for getting embeddings
with torch.inference_mode():
    points = model.get_embed(torch.tensor(encoded_text).unsqueeze(0))

print(points)

# Plotting the loss over epochs
plt.plot(range(1, len(tot_loss) + 1), tot_loss)
plt.scatter(range(1, len(tot_loss) + 1), tot_loss)
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
