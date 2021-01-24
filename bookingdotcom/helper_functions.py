import networkx as nx


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

