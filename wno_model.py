import torch
import torch.nn as nn
import numpy as np

class WaveletOperator(nn.Module):
    def __init__(self, seq_len, hidden_channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, 1, seq_len)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)          # (batch, 1, seq_len)
        return x[:, :, -1].squeeze()  # (batch,)

def train_wno_model(train_X, train_y, seq_len, hidden_channels=64, lr=1e-3, epochs=50, batch_size=32, device='cpu'):
    model = WaveletOperator(seq_len, hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_t = torch.tensor(train_X, dtype=torch.float32).to(device)
    y_t = torch.tensor(train_y, dtype=torch.float32).to(device)
    n = len(X_t)
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            Xb = X_t[batch_idx]
            yb = y_t[batch_idx]
            pred = model(Xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss: {total_loss/len(indices):.6f}")
    return model

def predict_wno(model, X):
    X_t = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(X_t)
        # out is a 1D tensor (batch,), we return a float if batch=1 else numpy array
        if out.numel() == 1:
            return float(out.item())
        else:
            return out.cpu().numpy()
