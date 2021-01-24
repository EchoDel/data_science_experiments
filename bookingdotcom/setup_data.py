import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import pickle as pkl
import torch

import bookingdotcom.helper_functions as helper_functions

# Load data
cache_location = Path('bookingdotcom/cache/')
training_data_path = Path('../data/bookingdotcom/training_dataset/booking_train_set.csv')

training_data = pd.read_csv(training_data_path, index_col=0)

training_data['checkin_date'] = pd.to_datetime(training_data['checkin'])
training_data['checkout_date'] = pd.to_datetime(training_data['checkout'])
training_data['days'] = training_data['checkout_date'] - training_data['checkin_date']


# Process Data

def trip_list(grouping: pd.Grouper):
    grouping = grouping.reset_index()
    output = {'city_id': [grouping['city_id'].to_list()],
              'countries': [grouping['hotel_country'].to_list()],
              'number_of_days': [[x.days for x in (grouping['days']).to_list()]],
              'starting_date': [grouping['checkin_date'].to_list()[0]],
              'user_id': [grouping['user_id'].to_list()[0]],
              }

    output = pd.DataFrame(output)

    return output


# group records into a trip
trip_dataframe = training_data.groupby('utrip_id').apply(trip_list)


trips = {}

# features to include, number of days at previous

for record in trip_dataframe.iterrows():
    trip_id = record[0][0]

    # response variable
    final_city = record[1]['city_id'][-1]
    current_city = record[1]['city_id'][-2]

    # get the previous cities in the trip as a single sparse tensor
    trip_previous_cities = record[1]['city_id'][:-1]
    trip_previous_cities_days = record[1]['number_of_days'][:-1]

    trip_cities = helper_functions.create_sparse_matrix(
        input_data = zip(trip_previous_cities,
                         [0] * len(trip_previous_cities),
                         trip_previous_cities_days),
        matrix_size = torch.Size([1, max(training_data['city_id']) + 1]))

    # get the previous cities visited by the users as a single sparse tensor
    user_id = record[1]['user_id']
    all_previous_cities = training_data[
        np.logical_and(training_data['user_id'] == user_id,
                       training_data['checkin_date'] < record[1]['starting_date'])]['city_id'].to_list()
    all_previous_cities_days = training_data[
        np.logical_and(training_data['user_id'] == user_id,
                       training_data['checkin_date'] < record[1]['starting_date'])]['days'].dt.days.to_list()

    previous_cities = helper_functions.create_sparse_matrix(
        input_data=zip(all_previous_cities,
                         [0] * len(all_previous_cities),
                         all_previous_cities_days),
        matrix_size=torch.Size([1, max(training_data['city_id']) + 1]))

    output = {'final_city': final_city,
              'trip_cities': trip_cities,
              'previous_cities': previous_cities,
              'current_city': current_city}
    trips[trip_id] = output

torch.save(trips, cache_location / 'trip_properties.pkl')
