import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Assuming TSMamba is a custom model, you would define or import it here
class TSMamba(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(TSMamba, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Custom Dataset for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Load and preprocess data
def load_data(filepath):
    data = pd.read_csv(filepath)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data['value'].values.reshape(-1, 1))
    return data_scaled, scaler

# Prepare DataLoader
def prepare_dataloader(data, seq_length, batch_size):
    dataset = TimeSeriesDataset(data, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Main script
if __name__ == "__main__":
    # Hyperparameters
    input_size = 1
    hidden_size = 50
    output_size = 1
    num_layers = 2
    seq_length = 10
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001

    # Load data
    data, scaler = load_data('your_time_series_data.csv')
    train_loader = prepare_dataloader(data, seq_length, batch_size)

    # Initialize model, loss function, and optimizer
    model = TSMamba(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Save the model
    torch.save(model.state_dict(), 'tsmamba_model.pth')

    # Load the model for inference (if needed)
    # model.load_state_dict(torch.load('tsmamba_model.pth'))
    # model.eval()
