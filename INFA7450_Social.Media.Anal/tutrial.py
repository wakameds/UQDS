#%%
import torch

class Graph:
    # Graph Construction
    def __init__(self, edges = None, weighted = True):
        # all nodes names
        node_set = set()
        for src, des, w in edges:
            node_set.add(src)
            node_set.add(des)
        self.node_names = list(node_set)

        self.weighted = weighted
        self.edges = edges
        self.node_number = len(self.node_names)

        self.Digraph = defaultdict(list)
        for node in self.node_names:
            self.Digraph[node]

        for src, des, w in edges:
            self.Digraph[src].append(des)

        self.name2index = defaultdict(int)
        self.index2name = defaultdict(str)
        for index, name in enumerate(self.Digraph.keys()):
            self.name2index[name] = index
            self.index2name[index] = name

        self.DiAdj = torch.zeros((self.node_number, self.node_number))
        for src, des, w in edges:
            self.DiAdj[self.name2index[src], self.name2index[des]] = w


    def weight(self, u_name, v_name):
        return self.DiAdj[self.name2index[u_name], self.name2index[v_name]]

    def has_node(self, n):
        return n in self.node_names

    def has_edge(self, u, v):
        return v in self.DiAdj[u]

    def graph_transpose(self):
        graph = defalutdict(list)
        for node in self.node_names:
            graph[node]

        for src, des, w in edges:
            graph[des].append(src)
        return graph

    def degree(self, u=None, mode ='out'):
        nodes = []
        if u is None:
            nodes = self.node_names
        else:
            nodes.append(u)
        degree = defaultdict(int)
        if mode == 'out':
            for node in nodes:
                degree[node] = len(self.Digraph[node])
            return degree
        elif mode == 'in':
            graph = self.graph_transpose()
            for node in nodes:
                degree[node] = len(graph[node])
            return degree
        else:
            graph = self
            graph.to_undirected()
            for node in nodes:
                degree[node] = len(graph[node])
            return degree
                




#%%
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

