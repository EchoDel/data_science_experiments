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
