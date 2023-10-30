import networkx as nx
import numpy as np
import random
from exponent_method import find_max_eigen_value
from exponent_method import find_max_eigen_value_scalar
from exponent_method import find_eigen_vector
from exponent_method import find_eigen_vector_equal_magnitude
from exponent_method import find_eigen_vector_auto


def max_indexes(arr, eps):
    abs_arr = np.abs(arr)
    max_value = np.max(abs_arr)
    return np.where(np.abs(abs_arr - max_value) < eps)[0] + 1


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


def generate_sample_graph():
    G = nx.DiGraph()
    G.add_nodes_from(range(1, 5))
    G.add_edge(1, 2)
    G.add_edge(2, 1)
    G.add_edge(1, 4)
    G.add_edge(4, 3)
    G.add_edge(3, 4)
    G.add_edge(2, 4)

    return G


def generate_graph(num):
    G = nx.DiGraph()

    # Create nodes with input and output edges
    for node in range(num):
        # Add the node to the graph
        G.add_node(node)

        # Add random input and output edges
        num_inputs = random.randint(1, num - 1)  # Random number of input edges (excluding self-loop)
        for _ in range(num_inputs):
            source = random.choice(range(num))  # Random source node
            if source != node and not G.has_edge(source, node):
                G.add_edge(source, node)

        num_outputs = random.randint(1, num - 1)  # Random number of output edges (excluding self-loop)
        for _ in range(num_outputs):
            target = random.choice(range(num))  # Random target node
            if target != node and not G.has_edge(node, target):
                G.add_edge(node, target)

    return G


def print_edges(G):
    print("Edges of the oriented graph:")
    for edge in G.edges():
        print(edge)
    return G


def perform(eps):
    G = generate_sample_graph()
    print_edges(G)
    mtx = to_matrix(G)
    print(mtx)
    lb2 = find_max_eigen_value_scalar(mtx, eps)
    print(lb2)
    x = find_eigen_vector_auto(mtx, eps)
    print(x)
    print("The most important nodes are:", max_indexes(x, eps))


def perform_generated(num, eps):
    G = generate_graph(num)
    print_edges(G)
    mtx = to_matrix(G)
    print(mtx)
    lb2 = find_max_eigen_value_scalar(mtx, eps)
    print("Max eigen value:", lb2)
    x = find_eigen_vector_auto(mtx, eps)
    print("Eigen vector:", x)
    print("The most important nodes are:", max_indexes(x, eps))
