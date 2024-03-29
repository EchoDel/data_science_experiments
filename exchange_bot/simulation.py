from datetime import datetime
from math import copysign, e

import pandas as pd
import torch


class ExchangeSimulation:
    '''
    State, data from the training_data parameter
    Reward, total money in gbp
    Done, When total money < half of the original
    '''
    def __init__(self, training_data: pd.DataFrame, starting_cash: float,
                 trading_cost: float, trading_percent_cost: float,
                 sell_percentage: float):
        self.starting_cash = starting_cash
        self.training_data = training_data
        self.trading_percent_cost = trading_percent_cost
        self.trading_cost = trading_cost
        self.sell_percentage = sell_percentage

        self.max_index = self.training_data.index.argmax()
        self.usd = 0
        self.eur = 0
        self.current_index = 0
        self.starting_index = 0
        self.steps = 0

        #Calculate the losses
        self.gain = 0
        self.sales_usd = 0

        self.reset()

    def current_date(self):
        return self.training_data[self.index].timestamp.to_list()[0]

    def reset(self, starting_point: datetime = None):
        self.usd = self.starting_cash
        self.eur = 0
        if starting_point is None:
            sample = self.training_data.sample(1)
        else:
            sample = self.training_data[self.training_data['timestamp'] ==
                                        starting_point]

        self.current_index = sample.index[0]
        self.starting_index = self.current_index
        self.gain = 0
        self.sales_usd = 0

    def reward(self):
        # Change to the the "lose" of the program
        return torch.Tensor([self.gain])

    def done(self):
        if self.current_index == self.max_index:
            return True
        elif self.starting_index + 525600 < self.current_index:
            return True
        return self.gain < -200

    def get_state(self):
        new_data = self.training_data.iloc[self.current_index].to_list()

        return new_data[1], \
            self.get_state_tensor([self.usd, self.eur] + new_data[0:1] + new_data[2:]), \
            self.reward(), \
            self.done()

    def get_state_tensor(self, state):
        return torch.Tensor([state])

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
        exchange_rate = self.training_data.iloc[self.current_index].Close
        if action == 0:  # buy
            current_usd = self.usd
            self.usd = current_usd * (1 - self.sell_percentage)
            sales_volume = current_usd * self.sell_percentage
            self.sales_usd += sales_volume
            self.eur = self.eur + sales_volume * exchange_rate * \
                (1 - self.trading_percent_cost) - self.trading_cost
        elif action == 2:  # sell
            exchange_rate = 1 / exchange_rate
            new_usd = self.eur * exchange_rate * \
                (1 - self.trading_percent_cost)

            self.gain += new_usd - self.sales_usd

            self.usd = self.usd + new_usd
            self.eur = 0
            self.sales_usd = 0
            self.usd = self.usd - self.trading_cost

        new_data = self.training_data.iloc[self.current_index].to_list()

        return new_data[1], \
            self.get_state_tensor([self.usd, self.eur] +
                                  new_data[0:1] +
                                  new_data[2:]), \
            self.reward(), \
            self.done()
