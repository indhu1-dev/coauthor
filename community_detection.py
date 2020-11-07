# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:47:32 2018

@author: Lenovo
"""

import community
import networkx as nx
import matplotlib.pyplot as plt

#better with karate_graph() as defined in networkx example.
#erdos renyi don't have true community structure
G = nx.read_adjlist("test.csv", delimiter=',', encoding='utf-8')
#first compute the best partition
print("Hello")

partition = community.best_partition(G)
print(partition.values)
#drawing
size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0.
print("II ")

for com in set(partition.values()) :
    print(" a ")

    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()