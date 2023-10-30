import networkx as nx
import numpy as np
import random
from exponent_method import find_max_eigen_value
from exponent_method import find_max_eigen_value_scalar
from exponent_method import find_eigen_vector
from exponent_method import find_eigen_vector_equal_magnitude
from exponent_method import find_eigen_vector_auto


def to_matrix(graph):
    num_nodes = nx.number_of_nodes(graph)
    matrix = np.zeros((num_nodes, num_nodes))
    for target_node in graph.nodes():
        outgoing_edges = list(graph.out_edges(target_node))
        if len(outgoing_edges) > 0:
            coefficient = 1.0 / len(outgoing_edges)
            for _, outgo in outgoing_edges:
                matrix[outgo - 1, target_node - 1] = coefficient
    return matrix


def generate_and_find(num_nodes, randomize=False):
    G = nx.DiGraph()
    G.add_nodes_from(range(1, num_nodes + 1))
    G.add_edge(1, 2)
    G.add_edge(2, 1)
    G.add_edge(1, 4)
    G.add_edge(4, 3)
    G.add_edge(3, 4)
    G.add_edge(2, 4)

    return G


def print_edges(G):
    print("Edges of the oriented graph:")
    for edge in G.edges():
        print(edge)
    return G


def perform(num,eps):
    G = generate_and_find(num)
    print_edges(G)
    mtx = to_matrix(G)
    print(mtx)
    lb2 = find_max_eigen_value_scalar(mtx, eps)
    print(lb2)
    x = find_eigen_vector_auto(mtx, eps)
    print(x)



