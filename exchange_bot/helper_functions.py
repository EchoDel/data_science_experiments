import torch
from matplotlib import pyplot as plt
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
        super(ExchangeBot, self).__init__()
        self.device = device

        self.features = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(relu_percentage),
            nn.Linear(input_size * 2, input_size * 4),
            nn.Dropout(),
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



def plot_durations(episode_durations, episode_reward):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    reward_t = torch.tensor(episode_reward, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    plt.plot(reward_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated