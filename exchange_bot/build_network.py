import math
from collections import deque, namedtuple
import random
from itertools import count

import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from exchange_bot.prepare_data import final_data
from exchange_bot.simulation import ExchangeSimulation
from exchange_bot.helper_functions import ExchangeBot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
STARTING_MONEY = 1000

n_actions = 3


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

simulation = ExchangeSimulation(final_data, STARTING_MONEY, 0, 0)
sample_simulation_return = simulation.get_state()


# Setup the networks
policy_net = ExchangeBot(input_size=len(sample_simulation_return[1]),
                         output_size=n_actions,
                         relu_percentage=0.3,
                         device=device).to(device)

target_net = ExchangeBot(input_size=len(sample_simulation_return[1]),
                         output_size=3,
                         relu_percentage=0.3,
                         device=device).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())


steps_done = 0


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
            return policy_net(state).argmax().view(1)
    else:
        return torch.tensor([random.randrange(n_actions)],
                            device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).view(
        (BATCH_SIZE, list(batch.state[0].shape)[0],))

    state_batch = torch.cat(batch.state, dim=0).view(
        (BATCH_SIZE, list(batch.state[0].shape)[0],))

    action_batch = torch.cat(batch.action).view(
        (BATCH_SIZE, list(batch.action[0].shape)[0],))

    reward_batch = torch.cat(batch.reward).view(
        (BATCH_SIZE, list(batch.action[0].shape)[0],))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros((BATCH_SIZE, n_actions), device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states)
    # Compute the expected Q values
    # GET REWARD BATH TO 128*3
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.float())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

memory = ReplayMemory(10000)
episode_durations = []



def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def get_state_tensor(state):
    state_tensor = state
    return state_tensor

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    simulation.reset()
    _, state, reward, done = simulation.get_state()
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
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
