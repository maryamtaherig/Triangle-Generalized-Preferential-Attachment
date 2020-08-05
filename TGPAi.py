from typing import Callable
import numpy as np
import networkx as nx
import scipy.io as sc
import random
import collections
from time import perf_counter
import matplotlib.pyplot as plt
import copy




MAX_ITERATIONS = 1000

#e_0 = np.ones((10,10)) - np.eye(10)

def TGPA_I(p_t: Callable, q_t: Callable) -> np.ndarray:
    """
    :param p_t:  probability function, takes int t as input
    :param q_t:  probability function, takes int t as input
    :return:     adjacency matrix
    """
    adjacency_matrix = np.zeros((3 * MAX_ITERATIONS , 3 * MAX_ITERATIONS ))

    # for i, e_r in enumerate(e_0):
    #     for j, a in enumerate(e_r):
    #         adjacency_matrix[i][j] = a
    list = []
    nodes_count = 0

    for t in range(1, MAX_ITERATIONS):

        matrix = adjacency_matrix[:]

        stages = [1, 2 ]
        stage = np.random.choice(stages, p=[p_t(t), q_t(t)])

        if stage == 1:  # run stage 1
            n_p = []
            v_t = nodes_count
            degrees = np.sum(adjacency_matrix, axis=0)
            # other nodes
            for i in range(nodes_count):
                pa = degrees[i] / (4 * t - 2)
                for _ in range(int(pa * 1000)):
                    n_p.append(i)
            # new node
            pa = 2 / (4 * t - 2)
            for _ in range(int(pa * 1000)):
                n_p.append(v_t)

            p = np.random.rand() * len(n_p)
            u = n_p[int(p)]
            adjacency_matrix[v_t][u] += 1
            adjacency_matrix[u][v_t] += 1

            n_p = []
            degree = np.sum(matrix[u])
            for n, a in enumerate(adjacency_matrix[u]):
                a = int(a)
                if a > 0 and n != u:
                    for _ in range(a):
                        n_p.append(n)
                if n == u:
                    for _ in range(2 * a):
                        n_p.append(n)

            p = np.random.rand() * len(n_p)
            w = n_p[int(p)]
            adjacency_matrix[v_t][w] += 1
            adjacency_matrix[w][v_t] += 1
            list.append(nodes_count)
            nodes_count += 1

        elif stage == 2:  # run stage 2:  # run stage 2
            adjacency_matrix[nodes_count][nodes_count + 1] = 1
            adjacency_matrix[nodes_count][nodes_count + 2] = 1
            adjacency_matrix[nodes_count + 1][nodes_count] = 1
            adjacency_matrix[nodes_count + 2][nodes_count] = 1
            list.append(nodes_count)
            nodes_count += 3
        m = 3
        if len(list) == m+1 :
            if list[-1] - list[0] > 3:
                for i in list(range(adjacency_matrix.shape[0]))[:-3]:
                    adjacency_matrix[i, -3] = sum[adjacency_matrix[i,-3:]]
                    # np.delete(adjacency_matrix, - , axis=0)
                adjacency_matrix = adjacency_matrix[:-3, :-3]
                adjacency_matrix[-1, :] = adjacency_matrix[:, -1]
                list.pop(0)
                nodes_count -= 2


    return adjacency_matrix

