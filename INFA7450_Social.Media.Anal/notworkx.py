import networkx as nx
G = nx.Graph()
eList = [('v1','v2'),
         ('v1','v3'),
         ('v2','v4'),
         ('v2','v3'),
         ('v3','v4'),
         ('v5','v4'),
         ('v5','v2'),
         ]

edges = [('s','t',10),
         ('s', 'y', 5),
         ('t', 'y', 2),
         ('t', 'x', 1),
         ('y', 't', 3),
         ('y', 'x', 9),
         ('y', 'z', 2),
         ('x', 'z', 4),
         ('z', 'x', 6),
         ('z', 's', 7)
         ]

G.add_edges_from(eList)
print(nx.degree_centrality(G))

#%%
G_eigen = nx.Graph()
G_eigen.add_edge('v1', 'v2')
G_eigen.add_edge('v2', 'v3')
print(nx.eigenvector_centrality(G_eigen))

#%%
print(nx.pagerank(G))
