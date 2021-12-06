import networkx as nx
import matplotlib.pyplot as plt
import os
os.chdir("/Users/wakayama.hideki/Desktop/UQ/03_Semester3/INFA7450_Social.Media.Anal/03_Assignment/01")

def EdgeList(file):
    edges = []
    with open(file, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            node1, _, node2 = line.partition(' ')
            node2, n = node2.split("\n")
            pairs = (node1, node2)
            edges.append(pairs)
    return edges

edges = EdgeList("3.data.txt")
g =nx.Graph(edges)

pos = nx.spring_layout(g)

#betweenness
bc_node10 = ['107','1684', '3437', '1912', '1085', '0', '698','567','58', '428']
coldict = {'107':"blue",
          '1684':"orange",
           '3437':"green",
           '1912':"red",
           '1085':"purple",
           '0':"brown",
           '698':"pink",
           '567':"gray",
           '58':"olive",
           '428':"cyan"
           }

bcnode_col = ["lightskyblue" for i in range(len(g.nodes))]

for i in bc_node10:
    bcnode_col[int(i)] = coldict[i]

bcnode_size = [10 for i in range(len(g.nodes))]
for i in bc_node10:
    bcnode_size[int(i)] = 200

bcedge_col = ["lightgrey" for i in range(len(g.edges))]
bcedges = [edge for edge in g.edges()]

plt.figure(figsize=(15,15))
nx.draw_networkx_nodes(g, pos, with_labels=False, node_color = bcnode_col, node_shape='.', node_size=bcnode_size)
nx.draw_networkx_edges(g, pos, edgelist=edges, edge_color=bcedge_col, width=0.1)
plt.axis("off")
plt.title('betweenness centrity')
plt.show()




#pagerank
pr_node10 = ['3437','107','1684','0','1912','348','686','3980','414','483']
coldictpr = {'3437':"blue",
          '107':"orange",
           '1684':"green",
           '0':"red",
           '1912':"purple",
           '348':"brown",
           '686':"pink",
           '3980':"gray",
           '414':"olive",
           '483':"cyan"
           }

prnode_col = ["lightskyblue" for i in range(len(g.nodes))]

for i in pr_node10:
    prnode_col[int(i)] = coldictpr[i]

prnode_size = [10 for i in range(len(g.nodes))]
for i in pr_node10:
    prnode_size[int(i)] = 200

predge_col = ["lightgrey" for i in range(len(g.edges))]
predges = [edge for edge in g.edges()]

plt.figure(figsize=(15,15))
nx.draw_networkx_nodes(g, pos, with_labels=False, node_color = prnode_col, node_shape='.', node_size=prnode_size)
nx.draw_networkx_edges(g, pos, edgelist=edges, edge_color=predge_col, width=0.1)
plt.axis("off")
plt.show()


