import torch
import torch.nn as nn


class LandmarkLSTMModel(nn.Module):
    def __init__(self, input_size=184, hidden_size=64, num_layers=3):
        super(LandmarkLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))
