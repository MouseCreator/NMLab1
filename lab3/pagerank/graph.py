import networkx as nx
import numpy as np
import random
from exponent_method import find_max_eigen_value
from exponent_method import find_max_eigen_value_scalar
from exponent_method import find_eigen_vector
from exponent_method import find_eigen_vector_equal_magnitude
from exponent_method import find_eigen_vector_auto
from exponent_method import find_max_eigen_value_auto


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
    graph = nx.DiGraph()

    for node in range(num):
        graph.add_node(node)

        num_inputs = random.randint(1, num - 1)
        for _ in range(num_inputs):
            source = random.choice(range(num))
            if source != node and not graph.has_edge(source, node):
                graph.add_edge(source, node)

        num_outputs = random.randint(1, num - 1)
        for _ in range(num_outputs):
            target = random.choice(range(num))
            if target != node and not graph.has_edge(node, target):
                graph.add_edge(node, target)

    return graph


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
    lb2 = find_max_eigen_value_auto(mtx, eps)
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


def modified(num, eps):
    G = generate_graph(num)
    print_edges(G)

    a_mtx = to_matrix(G)
    b_mtx = np.full((num, num), 1/num)
    alpha = 0.85

    m = a_mtx * alpha + b_mtx * (1 - alpha)

    print(m)
    lb2 = find_max_eigen_value_scalar(m, eps)
    print("Max eigen value:", lb2)
    x = find_eigen_vector_auto(m, eps)
    print("Eigen vector:", x)
    print("The most important nodes are:", max_indexes(x, eps))
