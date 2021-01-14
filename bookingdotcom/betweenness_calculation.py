import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import pickle as pkl

import bookingdotcom.helper_functions as helper_functions

network_location = Path('bookingdotcom/cache/network_graph.pkl')
save_location = Path('bookingdotcom/cache/betweeness.pkl')

booking_graph = nx.read_gpickle(network_location)

betweenness = nx.betweenness_centrality(booking_graph, 2000, weight="number_of_trips")

with open(save_location, mode='wb') as file:
    pkl.dump(betweenness, file)
