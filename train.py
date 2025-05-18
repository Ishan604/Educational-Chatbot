import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_util import tokenize, stem, bag_of_words
import numpy as np
import random
import os
from model import NeuralNet

# Load intents JSON file containing patterns and tags
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []  # List to hold all tokenized words from patterns
tags = []       # List to hold all unique tags (intents)
xy = []         # List to hold tuples of (tokenized pattern words, tag)

# Loop through each intent and its patterns to prepare training data
for intent in intents['intents']:
    tag = intent['tag']       # Get intent tag
    tags.append(tag)          # Collect tag for classification labels
    for pattern in intent['patterns']:
        w = tokenize(pattern)  # Tokenize pattern sentence into words
        all_words.extend(w)    # Add tokens to the all_words list
        xy.append((w, tag))    # Store pattern and its corresponding tag

# Preprocessing: ignore punctuation marks
ignore_words = ['?', '.', '!', ',']
# Stem all words and remove ignored punctuations
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Remove duplicates and sort the words alphabetically
all_words = sorted(set(all_words))
# Sort tags alphabetically
tags = sorted(set(tags))

# Prepare training data in bag-of-words format and corresponding labels
X_train = []
Y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)  # Convert pattern to bag-of-words vector
    X_train.append(bag)                              # Add vector to training data
    label = tags.index(tag)                          # Get label index for tag
    Y_train.append(label)                            # Add label to training labels

# Convert lists to numpy arrays for PyTorch compatibility
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Define a custom Dataset class for loading data during training
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)  # Number of samples
        self.x_data = X_train          # Input feature data
        self.y_data = Y_train          # Target labels

    def __getitem__(self, index):
        # Return input data as float tensor and label as tensor
        return torch.from_numpy(self.x_data[index]).float(), torch.tensor(self.y_data[index])

    def __len__(self):
        return self.n_samples  # Return total number of samples

# Hyperparameters for training
batch_size = 8
hidden_size = 16
output_size = len(tags)       # Number of intent classes
input_size = len(X_train[0])  # Size of input vector (bag-of-words length)
learning_rate = 0.001
num_epochs = 1000

# Create dataset and dataloader for batching and shuffling data
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Initialize the neural network model and move it to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Optional learning rate scheduler for adjusting the learning rate during training
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

# Training loop over epochs
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)                     # Move input to device
        labels = labels.to(dtype=torch.long).to(device)  # Move labels to device

        outputs = model(words)                       # Forward pass
        loss = criterion(outputs, labels)            # Calculate loss

        optimizer.zero_grad()                        # Zero gradients before backward pass
        loss.backward()                              # Backpropagate loss
        optimizer.step()                             # Update model parameters

    scheduler.step()                                 # Update learning rate per scheduler

    # Print loss every 100 epochs for monitoring
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Print final loss after training completes
print(f'Final Loss: {loss.item():.4f}')

# Save the trained model state and related metadata to a file for later use
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f"Training complete. File saved to {FILE}")
