#############################
# Here is my code for tasks #
#############################

# 1.Edge list from data
def EdgeList(file):
    edges = []
    with open(file, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            node1, _, node2 = line.partition(' ')
            node2, n = node2.split("\n")
            pairs = (node1, node2)
            edges.append(pairs)
    return edges


# 2. Class for making graph
from collections import defaultdict
import numpy as np
import math

class Graph:
    def __init__(self, edges = None):
        """construct graph
            :param
                edges([<str, str>,…,…]): edge list
            :returns
                node_names([str,str,…]): node set list
                edges(<str, str>): graph edges
                node_number(int): node number of the graph
                UndiGraph{node:list(str)}: Undirected graph dictionary
        """
        # node set list
        node_set = set()
        for src, des in edges:
            node_set.add(src)
            node_set.add(des)
        self.node_names = list(node_set)

        # graph edges
        self.edges = edges

        #graph node number
        self.node_number = len(self.node_names)

        #Undirected Graph
        self.UnDiGraph = defaultdict(list)
        for node, nei in self.edges:
            self.UnDiGraph[node].append(nei)
            self.UnDiGraph[nei].append(node)
        #Remove duplicated pairs
        for node, neighbours in self.UnDiGraph.items():
            self.UnDiGraph[node] = list(np.unique(neighbours))
        #Remove self-loop
        for node, neighbours in self.UnDiGraph.items():
            if node in neighbours:
                self.UnDiGraph[node] = neighbours.remove(node)

    def admat(self):
        """Create adjacency matrix
            return:array. value is 1 if node i links edge j. Otherwise, 0.
        """
        #create order list from 1 to the node number
        keyorder = []
        size = len(self.node_names)
        for num in range(len(self.node_names)):
            keyorder.append(str(num))

        #adjacency matrix
        A = [[0] * size for row in range(size)]
        for i in keyorder:
            for j in self.UnDiGraph[i]:
                A[int(i)][int(j)] = 1
        return np.array(A)

    def admatT(self):
        """ transpose adjacency matrix
            return: array
        """
        return self.admat().transpose()

    def degree(self, u = None):
        """ return degree of node u
            :param
                str: node u
            :return
                int: number of degree of node u
        """
        nodes = []
        if u is None:
            nodes = self.node_names
        else:
            nodes.append(u)
        degree = defaultdict(int)
        for node in nodes:
            degree[node] = len(self.UnDiGraph[node])
        return degree[u]

    def d_mat(self):
        """
        Create diagonal matrix with degree of graph nodes
            return: array
        """
        keyorder = []
        size = len(self.node_names)
        for num in range(len(self.node_names)):
            keyorder.append(str(num))

        D = [[0] * size for row in range(size)]
        for src in keyorder:
            j = int(src)
            D[int(src)][j] = self.degree(src)
            D = np.array(D)
        return D

    def dinvmat(self):
        """Create inverse diagonal matrix with degree of the nodes
            return:array
        """
        return np.linalg.inv(self.d_mat())


    def brandes_algorithm(self, V, E):
        """ perform brandes_algorithm
            param:
                V(list): nodes list
                E(list): edges list
            return:
                list: betweenness centrities of nodes
        """
        #Ininitl setting
        C = defaultdict(float)
        for src in V:
            dis = defaultdict(float) #distance
            sig = defaultdict(float) #sigma
            delta = defaultdict(float) #delta
            Pred = defaultdict(list) #

            for t in V:
                dis[t] = math.inf #set infnity
                sig[t] = 0 #source node

            dis[src] = 0
            sig[src] = 1
            Q = [src] #current node
            S = [] #visited node

            #Forward step
            while len(Q) > 0:
                v = Q.pop(0)
                S.append(v)
                #visit next node
                for w in E[v]:
                    if dis[w] == math.inf:
                        dis[w] = dis[v] + 1
                        Q.append(w)

                    if dis[w] == dis[v] + 1:
                        sig[w] = sig[w] + sig[v]
                        Pred[w].append(v)

            #Backward step
            for v in V:
                delta[v] = 0
                while len(S) > 0:
                    w = S.pop()
                    for v in Pred[w]:
                        delta[v] = delta[v] + (sig[v]/sig[w]) * (1 + delta[w])
                    if w != src:
                        C[w] = C[w] + delta[w]
        return C

    def PageRank1(self, AT, Dinv):
        """Performs PageRank calculation
            param:
                AT(array): transposed adjacent matrix
                Dinv(array): inverse diagonal matrix of node degree
            return:
                list: node PageRank score
        """
        alpha = 0.85
        beta = 0.15
        v = np.ones(len(self.node_names))
        M = AT @ Dinv
        I = np.identity(len(self.node_names))

        C = beta*np.linalg.inv((I - alpha*M)) @ v
        return C

    def PageRank2(self, AT, Dinv):
        """Performs PageRank calculation with iteration method
            param:
                AT(array): transposed adjacent matrix
                Dinv(array): inverse diagonal matrix of node degree
            return:
                list: node PageRank score
        """
        alpha = 0.85
        beta = 0.15
        maxloop = 100
        v = np.ones(len(self.node_names))
        I = np.ones(len(self.node_names))
        M = AT @ Dinv
        N = pow(10,-6)

        for i in range(maxloop):
            v_last = v
            v = alpha * M @ v_last + beta * I
            if abs(np.linalg.norm(v)-np.linalg.norm(v_last)) < N:
                return v
        return v





print('Here is your code for the tasks')

if __name__ == "__main__":
    #load data file
    edges = EdgeList("3.data.txt")
    #construct graph
    graph = Graph(edges)

    #Betweenness centrity
    BC_score = graph.brandes_algorithm(graph.node_names, graph.UnDiGraph)
    sort_BC = sorted(BC_score.items(), reverse=True, key=lambda x:x[1])
    print("Betweenness centrity top10: ", sort_BC[0:10])

    # 107: 7833120.28888149
    # 1684: 5506573.3738165535
    # 3437: 3849012.303142943
    # 1912: 3737836.424513493
    # 1085: 2429155.516720958
    # 0: 2384992.226158773
    # 698: 1880048.4929644058
    # 567: 1569993.811188233
    # 58: 1375189.966749343
    # 428: 1048328.1355515214

    #PageRank score
    PR_score = graph.PageRank2(graph.admatT(), graph.dinvmat())
    PR_dict = defaultdict(float)

    for i in range(len(PR_score)):
        PR_dict[i] = PR_score[i]
    sort_PR = sorted(PR_dict.items(), reverse=True, key = lambda x:x[1])
    print("PageRank top10: ", sort_PR[0:10])

    #PR
    #3437, 30.593680261957232
    #107, 27.82214680910343
    #1684, 25.479988369290616
    #0, 25.141554587040922
    #1912, 15.415045614701178
    #348, 9.359845516374854
    #686, 8.953622388310361
    #3980, 8.710315309522143
    #414, 7.198667522615942
    #483, 5.227143936413874

