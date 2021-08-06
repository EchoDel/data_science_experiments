from datetime import datetime
import pandas as pd
import torch


class ExchangeSimulation:
    '''
    State, data from the training_data parameter
    Reward, total money in gbp
    Done, When total money < half of the original
    '''
    def __init__(self, training_data: pd.DataFrame, starting_cash: float,
                 trading_cost: float, trading_percent_cost: float):
        self.starting_cash = starting_cash
        self.training_data = training_data
        self.trading_percent_cost = trading_percent_cost
        self.trading_cost = trading_cost

        self.gbp = 0
        self.eur = 0
        self.current_index = 0
        self.steps = 0

        self.reset()

    def current_date(self):
        return self.training_data[self.index].timestamp.to_list()[0]

    def reset(self, starting_point: datetime = None):
        self.gbp = self.starting_cash
        self.eur = 0
        if starting_point is None:
            sample = self.training_data.sample(1)
        else:
            sample = self.training_data[self.training_data['timestamp'] ==
                                        starting_point]

        self.current_index = sample.index[0]

    def reward(self):
        exchange_rate = self.training_data.iloc[self.current_index].Open
        return self.gbp + self.eur * (1 / exchange_rate)

    def done(self):
        return self.reward() < self.starting_cash / 2

    def get_state(self):
        new_data = self.training_data.iloc[self.current_index].to_list()

        return new_data[1], \
            self.get_state_tensor([self.gbp, self.eur] + new_data[0:1] + new_data[2:]), \
            self.reward(), \
            self.done()

    def get_state_tensor(self, state):
        return torch.Tensor(state)

    def step(self, action):
        '''
        :param action: 0 if buy, 1 if hold, 2 if sell
        :return: current timestamp,
        current state,
        reward,
         done
        '''
        self.current_index += 1
        self.steps += 1
        exchange_rate = self.training_data.iloc[self.current_index].Open
        if action == 0:
            current_gbp = self.gbp
            self.gbp = current_gbp / 2
            self.eur = self.eur + current_gbp / 2 * exchange_rate * \
                (1 - self.trading_percent_cost) - self.trading_cost
        elif action == 2:
            exchange_rate = 1 / exchange_rate
            self.gbp = self.gbp + self.eur * exchange_rate * \
                (1 - self.trading_percent_cost)
            self.eur = 0
            self.gbp = self.gbp - self.trading_cost

        new_data = self.training_data.iloc[self.current_index].to_list()

        return new_data[1], \
            self.get_state_tensor([self.gbp, self.eur] + new_data[0:1] + new_data[2:]), \
            self.reward(), \
            self.done()
