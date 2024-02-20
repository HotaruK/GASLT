import torch.nn as nn


class LandmarkLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LandmarkLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x
