import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Parallelize the apply functions
from pandarallel import pandarallel

pandarallel.initialize()

training_data_path = Path('../data/bookingdotcom/training_dataset/booking_train_set.csv')

training_data = pd.read_csv(training_data_path, index_col=0)


def trip_list(grouping: pd.Grouper):
    grouping = grouping.reset_index()
    output = {'city_id': [grouping['city_id'].to_list()],
              'countries': [grouping['hotel_country'].to_list()]}

    output = pd.DataFrame(output)

    return output


def last_city(grouping: pd.Grouper):
    grouping = grouping.reset_index()
    output = {'city_id': [grouping['city_id'].to_list()[-1]],
              'countries': [grouping['hotel_country'].to_list()[-1]]}

    output = pd.DataFrame(output)

    return output


test = training_data.groupby('utrip_id').parallel_apply(trip_list)

latest_city = training_data.groupby('utrip_id').parallel_apply(last_city).reset_index()


leaving_cities = latest_city.groupby(['city_id', 'countries']).agg({'utrip_id': 'count'})
all_cities = training_data.groupby(['city_id', 'hotel_country']).agg({'utrip_id': 'count'})



