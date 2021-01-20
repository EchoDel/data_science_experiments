import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import pickle as pkl

import bookingdotcom.helper_functions as helper_functions

save_location = Path('bookingdotcom/cache/')

booking_graph = nx.read_gpickle(save_location / 'network_graph.pkl')

# Betweenness
betweenness = nx.betweenness_centrality(booking_graph, 2000, weight="number_of_trips")

with open(save_location / 'betweenness.pkl', mode='wb') as file:
    pkl.dump(betweenness, file)

# Closeness
closeness = nx.closeness_centrality(booking_graph, distance="number_of_trips")

with open(save_location / 'closeness.pkl', mode='wb') as file:
    pkl.dump(closeness, file)

# Triangles, in theory more triangle means more people visit there
triangles = nx.triangles(booking_graph)

with open(save_location / 'triangles.pkl', mode='wb') as file:
    pkl.dump(triangles, file)


network_features = np.zeros((3, max(closeness.keys()) + 1))
for x in closeness.keys():
    network_features[0, x] = closeness[x]
    network_features[1, x] = betweenness[x]
    network_features[2, x] = triangles[x]

with open(save_location / 'network_features.pkl', mode='wb') as file:
    pkl.dump(network_features, file)
