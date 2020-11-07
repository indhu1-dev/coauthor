# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:08:32 2018

@author: Lenovo
"""

import community
import networkx as nx
import matplotlib.pyplot as plt

#better with karate_graph() as defined in networkx example.
#erdos renyi don't have true community structure
G = nx.read_adjlist("test.csv", delimiter=',', encoding='utf-8')
#first compute the best partition
partition = community.best_partition(G)

#drawing
size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()


def _apply_prediction(G, func, ebunch=None):
   
    if ebunch is None:
        ebunch = nx.non_edges(G)
    return ((u, v, func(u, v)) for u, v in ebunch)




def cn_soundarajan_hopcroft(G, ebunch=None, community='community'):
   
    def predict(u, v):
        Cu = _community(G, u, community)
        Cv = _community(G, v, community)
        cnbors = list(nx.common_neighbors(G, u, v))
        neighbors = (sum(_community(G, w, community) == Cu for w in cnbors)
                     if Cu == Cv else 0)
        return len(cnbors) + neighbors
    return _apply_prediction(G, predict, ebunch)




def ra_index_soundarajan_hopcroft(G, ebunch=None, community='community'):
  
    def predict(u, v):
        Cu = _community(G, u, community)
        Cv = _community(G, v, community)
        if Cu != Cv:
            return 0
        cnbors = nx.common_neighbors(G, u, v)
        return sum(1 / G.degree(w) for w in cnbors
                   if _community(G, w, community) == Cu)
    return _apply_prediction(G, predict, ebunch)




def within_inter_cluster(G, ebunch=None, delta=0.001, community='community'):
    
    if delta <= 0:
        raise nx.NetworkXAlgorithmError('Delta must be greater than zero')

    def predict(u, v):
        Cu = _community(G, u, community)
        Cv = _community(G, v, community)
        if Cu != Cv:
            return 0
        cnbors = set(nx.common_neighbors(G, u, v))
        within = set(w for w in cnbors
                     if _community(G, w, community) == Cu)
        inter = cnbors - within
        return len(within) / (len(inter) + delta)

    return _apply_prediction(G, predict, ebunch)



def _community(G, u, community):
    """Get the community of the given node."""
    node_u = G.nodes[u]
    try:
        return node_u[community]
    except KeyError:
        raise nx.NetworkXAlgorithmError('No community information')
        def main():
            b = cn_soundarajan_hopcroft(G,community=partition)
            c = ra_index_soundarajan_hopcroft(G,community=partition)
            d = within_inter_cluster(G,community=partition)
            for j in b:
                        print (j)
                        for k in c:
                            print (k)
                        for l in d:
                            print (l)
                            if __name__ == "__main__":
                                main()
