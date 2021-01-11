import networkx as nx


def random_subgraph(graph: nx.Graph):
    ending_nodes = []
    selecting_nodes = []
    for node in nx.bfs_edges(graph, 2):
        if node[0] not in ending_nodes:
            ending_nodes.append(node[0])
            if len(ending_nodes) == 4:
                break
        selecting_nodes.append(node[1])

    k = graph.subgraph(selecting_nodes)

    return k