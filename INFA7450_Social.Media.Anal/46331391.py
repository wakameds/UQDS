from collections import defaultdict
import numpy as np
from gensim.models import Word2Vec
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
import networkx as nx

##Data preparation
def readTestdata(filename):
    testData = []
    with open(filename) as f:
        for line in f:
            a1, a2 = line.strip().split()
            testData.append((a1,a2))
    return testData

def readValidationdata(file_pos, file_neg):
    validation_data_pos = []
    validation_data_neg = []
    validationData = []
    with open(file_pos) as f:
        for line in f:
            a1, a2 = line.strip().split()
            validation_data_pos.append((a1,a2))
            validationData.append((a1,a2))
    with open(file_neg) as f:
        for line in f:
            a1, a2 = line.strip().split()
            validation_data_neg.append((a1,a2))
            validationData.append((a1,a2))
    return validationData, validation_data_pos, validation_data_neg


def readTrainData(filename):
    trainingData =[]
    with open(filename) as f:
        for line in f:
            a1, a2 = line.strip().split()
            trainingData.append((a1,a2))
    return trainingData


def getNeighbours(data):
    """
    prepare the dictionary with the neighbours
    :param] <list> data: edges list
    :return: <dict: node: neighbours> neighbours list with the node
    """
    neighbours = defaultdict(set)
    for pair in data:
        neighbours[pair[0]].add(pair[1])
        neighbours[pair[1]].add(pair[0])
    return neighbours

#Neighbourhood-base method
def computeProximityScore(measure, node1, node2, neighbors):
    """
    calculation score between the nodes
    :param measure: <str: menthods>
    :param node1: <str: node>
    :param node2: <set: node>
    :param neighbors: <dict: node:neighbours>
    :return:
    """
    score = 0
    if measure == 'Jaccard':
        score = len(neighbors[node1].intersection(neighbors[node2]))
        score = float(score)/len(neighbors[node1].union(neighbors[node2]))

    elif measure == 'Cosine':
        if np.sqrt(len(neighbors[node1]) * len(neighbors[node2])) > 0:
            score = len(neighbors[node1].intersection(neighbors[node2]))
            score = float(score)/np.sqrt(len(neighbors[node1])*len(neighbors[node2]))

    elif measure == 'Preferential':
        score = len(neighbors[node1])*len(neighbors[node2])

    elif measure == 'Common':
        score = len(neighbors[node1].intersection(neighbors[node2]))

    elif measure == 'Adamic':
        commonneis = neighbors[node1].intersection(neighbors[node2])
        for nei in commonneis:
            score = score + 1/np.log(len(neighbors[nei]))
    return score


#Random walk based method
def buildG(data):
    G = nx.Graph()
    for pair in data:
        G.add_edge(pair[0], pair[1])
    return G


def BiasedRW(G, walklength):
    """
    random walk sampling
    :param G: Graph
    :param walklength:<int:walk length>
    :return: list
    """
    rw = BiasedRandomWalk(G)
    walks = rw.run(nodes = G.nodes(), length=walklength,
                   n=5, p=0.3, q=0.1, weighted=False, seed=123)
    return walks


def Model(walks):
    model = Word2Vec(walks, window=8, sg=1)
    return model


def computationScoreDW(em1, em2, measure):
    if measure == 'Cossine':
        score = np.dot(em1,em2)/(np.linalg.norm(em1)*np.linalg.norm(em2))
    elif measure == 'Euclidean':
        score = 1/np.linalg.norm(em1 - em2)
    elif measure == 'Dot':
        score = np.dot(em1,em2)
    return score


def top100List_N2V(dataset, model):
    """
    output the top100 prediction links
    :param dataset: list[(node,node)]
    :param model: model
    """
    Scores = dict()
    for pair in dataset:
        Scores[pair[0]+' '+pair[1]] = computationScoreDW(model.wv[pair[0]], model.wv[pair[1]], 'Cossine')
    top100Links = dict(sorted(Scores.items(), key=lambda x: x[1], reverse=True)[:100])
    return top100Links


def evaluation(topList, groundTruth):
    count=0
    for link in topList:
        if link in groundTruth:
            count += 1
    return (float(count)/len(groundTruth)*100, '%')


if __name__ == '__main__':
    trainset = readTrainData('training.txt')
    valset, valposset, valnegset = readValidationdata('val_positive.txt', 'val_negative.txt')
    testset = readTestdata('test.txt')


    neighbours = getNeighbours(trainset)
    groundTruth = {}.fromkeys([pair[0] + ' ' + pair[1] for pair in valposset])

    #Neighbourhood-base
    linkScores_JC ={}
    linkScores_CS ={}
    linkScores_AD = {}
    linkScores_PR = {}
    linkScores_CM = {}
    for pair in valset:
        linkScores_JC[pair[0]+' '+pair[1]] = computeProximityScore('Jaccard', pair[0], pair[1], neighbours)
        linkScores_CS[pair[0]+' '+pair[1]] = computeProximityScore('Cosine', pair[0], pair[1], neighbours)
        linkScores_AD[pair[0] + ' ' + pair[1]] = computeProximityScore('Adamic', pair[0], pair[1], neighbours)
        linkScores_PR[pair[0] + ' ' + pair[1]] = computeProximityScore('Preferential', pair[0], pair[1], neighbours)
        linkScores_CM[pair[0] + ' ' + pair[1]] = computeProximityScore('Common', pair[0], pair[1], neighbours)

    top100Links_Jaccard = dict(sorted(linkScores_JC.items(), key=lambda x:x[1], reverse=True)[:100])
    top100Links_Cosine = dict(sorted(linkScores_CS.items(), key=lambda x: x[1], reverse=True)[:100])
    top100Links_Adamic = dict(sorted(linkScores_AD.items(), key=lambda x: x[1], reverse=True)[:100])
    top100Links_Preferential = dict(sorted(linkScores_PR.items(), key=lambda x: x[1], reverse=True)[:100])
    top100Links_Common = dict(sorted(linkScores_CM.items(), key=lambda x: x[1], reverse=True)[:100])

    #Node2Vec
    G = buildG(trainset)
    SG = StellarGraph.from_networkx(G)

    groundTruth = {}.fromkeys([pair[0] + ' ' + pair[1] for pair in valposset])

    walklength = 10

    walks = BiasedRW(SG, walklength)
    model = Model(walks)
    top100Links_N2V_val = top100List_N2V(valset, model)
    evaluation(top100Links_N2V_val, groundTruth)


    print('JC:{}'.format(evaluation(top100Links_Jaccard, groundTruth)),
          'Cosine{}'.format(evaluation(top100Links_Cosine, groundTruth)),
          'Adamic{}'.format(evaluation(top100Links_Adamic, groundTruth)),
          'Preferential{}'.format(evaluation(top100Links_Preferential, groundTruth)),
          'Common{}'.format(evaluation(top100Links_Common, groundTruth)),
          'Node2Vec{}'.format(evaluation(top100Links_N2V_val, groundTruth))
          )


    #Test dataset
    #Neighbourhood-base
    linkScores_JC_test ={}
    linkScores_CS_test ={}
    linkScores_AD_test = {}
    linkScores_PR_test = {}
    for pair in testset:
        linkScores_JC_test[pair[0]+' '+pair[1]] = computeProximityScore('Jaccard', pair[0], pair[1], neighbours)
        linkScores_CS_test[pair[0]+' '+pair[1]] = computeProximityScore('Cosine', pair[0], pair[1], neighbours)
        linkScores_AD_test[pair[0] + ' ' + pair[1]] = computeProximityScore('Adamic', pair[0], pair[1], neighbours)
        linkScores_PR_test[pair[0] + ' ' + pair[1]] = computeProximityScore('Preferential', pair[0], pair[1], neighbours)

    top100Links_Jaccard_test = dict(sorted(linkScores_JC_test.items(), key=lambda x:x[1], reverse=True)[:100])
    top100Links_Cosine_test = dict(sorted(linkScores_CS_test.items(), key=lambda x: x[1], reverse=True)[:100])
    top100Links_Adamic_test = dict(sorted(linkScores_AD_test.items(), key=lambda x: x[1], reverse=True)[:100])
    top100Links_Preferential_test = dict(sorted(linkScores_PR_test.items(), key=lambda x: x[1], reverse=True)[:100])

    #Node2Vec
    G = buildG(testset)
    SG = StellarGraph.from_networkx(G)
    walklength = 20

    walks = BiasedRW(SG, walklength)
    model = Model(walks)
    top100Links_N2V_test = top100List_N2V(testset, model)



#%%
neighbours_count ={}
for node, nei in neighbours.items():
    neighbours_count[node]=len(nei)

x = neighbours_count.values()
import matplotlib.pyplot as plt
plt.hist(x)
plt.xlabel('the number of neighbours')
plt.ylabel('the number of authors')
plt.show()



#%%
results = [pair + '\n' for pair in top100Links_Adamic_test.keys()]
with open('results_Adamic.txt', 'w') as f:
    f.writelines(results)