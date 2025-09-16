import networkx as nx


def build_supplier_graph(nodes=None, edges=None):
    G = nx.Graph()
    nodes = nodes or []
    edges = edges or []
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G
