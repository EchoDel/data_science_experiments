import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx

import bookingdotcom.helper_functions as helper_functions

save_location = Path('bookingdotcom/cache/network_graph.pkl')

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
    ug.add_node(x[1]['city_id'])
    ug.nodes[x[1]['city_id']]['country'] = x[1]['hotel_country']
    ug.nodes[x[1]['city_id']]['number_of_trips'] = x[1]['utrip_id']

for x in edges:
    ug.add_edge(x[0], x[1], weight=edges[x])


for x in last_cities:
    ug.nodes[x]['trip_finishes'] = last_cities[x]

# Save graph
save_location.parent.mkdir(parents=True, exist_ok=True)
nx.write_gpickle(ug, save_location)



k = helper_functions.random_subgraph(ug, depth=20, starting_node=2)

colour_map = []
for x in k:
    if 'trip_finishes' in k.nodes[x]:
        if k.nodes[x]['trip_finishes'] > 20:
            colour_map.append('green')
        elif k.nodes[x]['trip_finishes'] > 10:
            colour_map.append('yellow')
        else:
            colour_map.append('blue')
    else:
        colour_map.append('red')

nx.draw(k, with_labels=True, node_color=colour_map)
