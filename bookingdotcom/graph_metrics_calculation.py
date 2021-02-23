import numpy as np
from pathlib import Path
import networkx as nx
import pickle as pkl
import torch

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


# setup sparse matrices of the parameters for each city/node
connected_node_features = {}
for node in booking_graph.nodes():
    connected_nodes = [x[1] for x in booking_graph.edges(node)]
    xs = []
    ys = []
    values = []

    for value_node in connected_nodes:
        ys += ([value_node] * 3)
        xs += [0, 1, 2]
        values += network_features[:, value_node].tolist()

    i = torch.LongTensor([xs,
                          ys])
    v = torch.FloatTensor(values)
    connected_node_features[node] = torch.sparse.FloatTensor(i,
                                                             v,
                                                             torch.Size([3, max(booking_graph.nodes()) + 1]))

torch.save(connected_node_features, save_location / 'connected_node_features.pkl')
