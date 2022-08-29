from collections import deque, namedtuple
import random

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Optimizer


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
                 dropout_percentage: float, device: str):
        super(ExchangeBot, self).__init__()
        self.device = device

        self.features = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(),
            nn.Linear(input_size * 2, input_size * 4),
            nn.Dropout(dropout_percentage),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_size * 4, input_size * 2),
            nn.ReLU(),
            nn.Linear(input_size * 2, output_size),
        )

    def forward(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        feature = self.features(input_tensor)
        return self.classifier(feature)


def plot_durations(episode_durations, episode_reward, training_graph_location):
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)

    ax2 = ax.twinx()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    reward_t = torch.tensor(episode_reward, dtype=torch.float)
    ax.title.set_text('Training...')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Duration')
    ax.plot(durations_t.numpy())
    ax2.plot(reward_t.numpy(), c='y')
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax.plot(means.numpy())

    fig.savefig(training_graph_location / f'training_{len(durations_t)}')


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model(memory: ReplayMemory, BATCH_SIZE: int,
                   device: str, GAMMA: float, optimizer: Optimizer,
                   policy_net: ExchangeBot, target_net: ExchangeBot):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    batch_size = len(batch.state)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
