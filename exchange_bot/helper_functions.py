import torch
from torch import nn


class ExchangeBot(nn.Module):
    '''
    Takes input of a vector of the current state of the exchange rate which is

    1 Current exchange rate
    2-41 EMA(1-40 days)
    42 Number of current stocks
    43 Buy prices

    Output
    Vector of buy, hold, and sell probabilities
    '''

    def __init__(self, input_size: int, output_size: int,
                 relu_percentage: float, device: str):
        super(ExchangeBot).__init__()
        self.device = device

        self.features = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(relu_percentage),
            nn.Linear(input_size * 2, input_size * 4),
            nn.ReLU(relu_percentage),
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_size * 4, input_size * 2),
            nn.ReLU(relu_percentage),
            nn.Linear(input_size * 2, output_size),
        )

    def forward(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        feature = self.features(input_tensor)
        return self.classifier(feature)
