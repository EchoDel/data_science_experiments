import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import pickle as pkl

import bookingdotcom.helper_functions as helper_functions

network_location = Path('bookingdotcom/cache/network_graph.pkl')
save_location = Path('bookingdotcom/cache/')

booking_graph = nx.read_gpickle(network_location)

# Betweenness
betweenness = nx.betweenness_centrality(booking_graph, 2000, weight="number_of_trips")

with open(save_location / 'betweeness.pkl', mode='wb') as file:
    pkl.dump(betweenness, file)

# Closeness
closeness = nx.closeness_centrality(booking_graph, distance="number_of_trips")

with open(save_location / 'closeness.pkl', mode='wb') as file:
    pkl.dump(closeness, file)

# Triangles, in theory more triangle means more people visit there
closeness = nx.triangles(booking_graph)

with open(save_location / 'triangles.pkl', mode='wb') as file:
    pkl.dump(closeness, file)
