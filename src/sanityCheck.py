import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ChessPrepLib import convertFEN, revertFEN, encode_custom_fen, unencode_custom_fen

import torch
from PyLinq import PyLinqData
# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device("cuda")
else:
    print("CUDA is not available")
    device = torch.device("cpu")

# X = np.random.rand(n_samples, n_features).astype(np.float32)  # Random float vectors
# y = X.copy()  # Identity function, output is the same as input
# instead of random data, we will use the data from the csv file and process it a bit
# avoid pandas for now only use numpy.  Get X_raw from the first column of the csv file, and y_raw from the second column
import csv
def load_chess_data():
    X_raw = None
    y_raw = None
    with open('../X_training_Y_Labels_Comment.csv', 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        # since the csv contains two columns of strings, extract each column into X_raw and y_raw lists of strings
        X_raw = [row[0] for row in data]
        y_raw = [row[1] for row in data]
    # convert each string X_raw and y_raw to the custom FEN string format using the convertFEN function
    X_conv = [convertFEN(fen) for fen in X_raw]
    y_conv = [convertFEN(fen) for fen in y_raw]
    # convert the custom FEN string format to a numpy array using the encode_custom_fen function
    # for each element in X_conv and y_conv convert to a numpy array of floats and append to X and y
    X = np.array([encode_custom_fen(fen) for fen in X_conv], dtype=np.float32)
    y = np.array([encode_custom_fen(fen) for fen in y_conv], dtype=np.float32)
    return X, y

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Task selection
task = input("Enter 'identity' for identity function learning or 'chess' for chess move prediction: ").strip().lower()
# prompt for the number of epochs
epochs = int(input("Enter the number of epochs: ").strip())
model_filename = "identityModel.pt" if task == "identity" else "chessModel.pt"

# Data Loading and Preprocessing
n_samples = 52000
n_features = 69

if task == "identity":
    # Generate synthetic data for identity function
    X = np.random.rand(n_samples, n_features).astype(np.float32)  # Random float vectors
    y = X.copy()  # Output is the same as input for identity function
else:
    # Load your chess data here for the chess move prediction task
    # This is a placeholder; you will replace it with actual data loading code
    X, y = load_chess_data()  # Implement this function based on your data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Neural Network Definition
class CustomNet(nn.Module):
    def __init__(self, n_features, hidden_dim):        
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, n_features)
        # attempt load the model from the file model_filename using try except
        try:
            self.load_state_dict(torch.load(model_filename))
            print("Model loaded from", model_filename)
        except:
            print("Model not found, creating a new model")

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

hidden_dim = 100  # Tweakable parameter
model = CustomNet(n_features, hidden_dim)
model.to(device)

# Loss, Optimizer, and Scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Training Loop
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), model_filename)
print("Training complete. Model saved as", model_filename)
