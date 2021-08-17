import math
from collections import namedtuple
import random
from itertools import count
from pathlib import Path

import progressbar

import torch
from torch import optim, nn
from exchange_bot.prepare_data import final_data
from exchange_bot.simulation import ExchangeSimulation
from exchange_bot.helper_functions import ExchangeBot, plot_durations, \
    ReplayMemory, optimize_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
STARTING_MONEY = 1000
SELL_PERCENTAGE = 0.1

training_graph_location = Path('cache/performance_graphs')
n_actions = 3

training_graph_location.mkdir(exist_ok=True, parents=True)

simulation = ExchangeSimulation(final_data, STARTING_MONEY,
                                0, 0, SELL_PERCENTAGE)
sample_simulation_return = simulation.get_state()


# Setup the networks
policy_net = ExchangeBot(input_size=sample_simulation_return[1].shape[1],
                         output_size=n_actions,
                         dropout_percentage=0.3,
                         device=device).to(device)

target_net = ExchangeBot(input_size=sample_simulation_return[1].shape[1],
                         output_size=3,
                         dropout_percentage=0.3,
                         device=device).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device,
                            dtype=torch.long)


memory = ReplayMemory(1000)
episode_durations = []
episode_reward = []

widgets = [
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(marker=progressbar.RotatingMarker()),
    'Interations:', progressbar.Counter(), '   ',
    progressbar.Variable('reward', precision=5),
    progressbar.Variable('episodes', precision=2)
]

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    simulation.reset()
    _, state, reward, done = simulation.get_state()
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength,
                                  variables={'episodes': i_episode,
                                             'reward': reward.item(), },
                                  widgets=widgets)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, state, reward, done = simulation.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = state
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the policy network)
        optimize_model(memory, BATCH_SIZE, device, GAMMA, optimizer,
                       policy_net, target_net)
        if done:
            episode_durations.append(t + 1)
            episode_reward.append(reward.item())
            plot_durations(episode_durations, episode_reward,
                           training_graph_location)
            break
        bar.update(t, reward=reward.item())
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
