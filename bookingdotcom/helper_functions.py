import networkx as nx
import torch
import numpy as np
import random as rand
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F


def random_subgraph(graph: nx.Graph, depth=4, starting_node=2):
    ending_nodes = []
    selecting_nodes = []
    for node in nx.bfs_edges(graph, starting_node):
        if node[0] not in ending_nodes:
            ending_nodes.append(node[0])
            if len(ending_nodes) == depth:
                break
        selecting_nodes.append(node[1])

    k = graph.subgraph(selecting_nodes)

    return k


def create_sparse_matrix(input_data: zip, matrix_size: torch.Size):
    xs = []
    ys = []
    values = []
    # previous cities in the trip
    for value_node in input_data:
        ys += [value_node[0]]
        xs += [value_node[1]]
        values += [value_node[2]]

    i = torch.LongTensor([xs,
                          ys])
    v = torch.FloatTensor(values)

    sparse_matrix = torch.sparse.FloatTensor(i,
                                             v,
                                             matrix_size)

    return sparse_matrix


class BookingLoader(torch.utils.data.Dataset):
    def __init__(self, trips, connected_node_features, training, training_percentage, number_of_classes, seed=1994):
        super(BookingLoader).__init__()
        self.trips = trips
        self.connected_node_features = connected_node_features
        self.training = training
        self.training_percentage = training_percentage
        self.number_of_classes = number_of_classes

        rand.seed(seed)
        np.random.seed(seed)

        self.k = int(round(len(self.trips.keys()) * self.training_percentage))
        self.indices = rand.sample(self.trips.keys(), self.k)
        if not self.training:
            self.indices = [x for x in trips.keys() if x not in self.indices]

        self.sample_trips = [self.trips[index] for index in self.indices]
        self.n = 0
        self.start = 0
        self.end = len(self.indices)

    def get_one_hot(self, target):
        a = torch.zeros(self.number_of_classes, dtype=torch.float)
        a[target] = 1
        return a

    def load_sample(self, index):
        dict_key = self.indices[index]
        final_city = self.trips[dict_key]['final_city']

        if self.trips[dict_key]['current_city'] == 0:
            self.trips[dict_key]['current_city'] = 2

        connected_node_features = self.connected_node_features[self.trips[dict_key]['current_city']]
        trip_cities = self.trips[dict_key]['trip_cities']
        previous_cities = self.trips[dict_key]['previous_cities']

        return self.get_one_hot(final_city), trip_cities, previous_cities, connected_node_features, dict_key

    def __next__(self):
        if self.n < self.end:
            n = self.n
            final_city, trip_cities, previous_cities, connected_node_features, trip_id = self.__getitem__(n)
            self.n += 1
            return final_city, trip_cities, previous_cities, connected_node_features, trip_id
        else:
            self.n = 0
            raise StopIteration

    def __getitem__(self, index):
        final_city, trip_cities, previous_cities, connected_node_features, trip_id  = self.load_sample(index)
        return final_city, trip_cities, previous_cities, connected_node_features, trip_id

    def __len__(self):
        return self.end




class LinearNN(nn.Module):
    def __init__(self, city_numbers) -> None:
        super(LinearNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(city_numbers, 2048),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(2048, 4096),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 4096),
        )

        self.features = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.3),
        )

        self.classifier = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, city_numbers),
        )

    def forward(self,
                closeness: torch.Tensor,
                betweenness: torch.Tensor,
                triangles: torch.Tensor,
                trip_cities: torch.Tensor,
                previous_cities: torch.Tensor,) -> torch.Tensor:
        closeness_fc = self.fc(closeness)
        betweenness_fc = self.fc(betweenness)
        triangles_fc = self.fc(triangles)
        trip_cities_fc = self.fc(trip_cities)
        previous_cities_fc = self.fc(previous_cities)

        x = closeness_fc + betweenness_fc + triangles_fc + trip_cities_fc + previous_cities_fc
        x = self.features(x)
        x = self.classifier(x)
        return x
