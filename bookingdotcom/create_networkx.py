import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx

import bookingdotcom.helper_functions as helper_functions


# Initialize Graph
ug = nx.Graph(directed=False)

training_data_path = Path('../data/bookingdotcom/training_dataset/booking_train_set.csv')

training_data = pd.read_csv(training_data_path, index_col=0)

# Nodes are cities
nodes = training_data.groupby(['city_id', 'hotel_country']).agg({'utrip_id': 'count'}).reset_index()


# Edges are connecting cities by trip with a weighting of how often they are selected
last_trip = '0'
last_city = ''
edges = {}
last_cities = {}

for record in training_data.iterrows():
    if last_trip == record[1]['utrip_id']:
        index = tuple(sorted([record[1]['city_id'], last_city]))
        if index in edges:
            edges[index] += 1
        else:
            edges[index] = 1
    else:
        if record[1]['city_id'] in last_cities:
            last_cities[record[1]['city_id']] += 1
        else:
            last_cities[record[1]['city_id']] = 1
        last_trip = record[1]['utrip_id']
        last_city = record[1]['city_id']


for x in nodes.iterrows():
    ug.add_node(x[1]['utrip_id'],
                country=x[1]['hotel_country'],
                number_of_trips=x[1]['utrip_id'])

for x in edges:
    ug.add_edge(x[0], x[1], weight=edges[x])


for x in last_cities:
    ug.nodes[x]['trip_finishes'] = last_cities[x]

