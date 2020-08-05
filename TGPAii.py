from typing import Callable
import numpy as np
import networkx as nx
import scipy.io as sc
import random
import collections
from time import perf_counter
import matplotlib.pyplot as plt
import warnings
import TGPAi

warnings.filterwarnings("ignore")
MAX_ITERATIONS = 10000

e_0 = np.ones((10,10)) - np.eye(10)

def TGPA_II(e_0: np.ndarray, p_t: Callable, r_t: Callable, q_t: Callable) -> np.ndarray:
    """
    :param e_0:  input adjacency matrix
    :param p_t:  probability function, takes int t as input
    :param r_t:  probability function, takes int t as input
    :param q_t:  probability function, takes int t as input
    :return:     adjacency matrix
    """
    adjacency_matrix = np.zeros((3 * MAX_ITERATIONS + len(e_0), 3 * MAX_ITERATIONS + len(e_0)))

    for i, e_r in enumerate(e_0):
        for j, a in enumerate(e_r):
            adjacency_matrix[i][j] = a

    nodes_count = len(e_0)

    for t in range(1, MAX_ITERATIONS):

        matrix = adjacency_matrix[:]

        stages = [1, 2 ,3]#3
        stage = np.random.choice(stages, p=[p_t(t), r_t(t), q_t(t)])

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

            nodes_count += 1
        elif stage == 2:  # run stage 2
            n_p = []
            degrees_sum = np.sum(np.sum(matrix, axis=0))
            degrees = np.sum(matrix, axis=0)
            for i in range(nodes_count):
                pa = degrees[i] / degrees_sum
                for _ in range(int(pa * 1000)):
                    n_p.append(i)

            p = np.random.rand() * len(n_p)
            v1 = n_p[int(p)]

            p = np.random.rand() * len(n_p)
            v2 = n_p[int(p)]

            adjacency_matrix[v1][v2] += 1
            adjacency_matrix[v2][v1] += 1
        else:  # run stage 3
            adjacency_matrix[nodes_count][nodes_count + 1] = 1
            adjacency_matrix[nodes_count][nodes_count + 2] = 1
            adjacency_matrix[nodes_count + 1][nodes_count] = 1
            adjacency_matrix[nodes_count + 2][nodes_count] = 1
            nodes_count += 3

    return adjacency_matrix


# matrix = TGPA_II([[1, 1], [1, 1]], lambda t: 1/3, lambda t: 1/3, lambda t: 1/3)
# for i, m in enumerate(matrix):
#     for j, cell in enumerate(m):
#         print(cell, ',', end='')
#     print('\n')


def main():
    t = []
    dataset_names = ['Princeton12', 'Berkeley13', 'Auburn71']
    dataset_name = dataset_names[0]
    # ----- load Graph -----
    print("Loading Graph ....")
    test = sc.loadmat(dataset_name + '.mat')
    # TGPAi.TGPA_I(lambda t: 0.99, lambda t: 0.01)
    A = TGPAi.TGPA_I(lambda t: 0.99, lambda t: 0.01)
#TGPA_II(e_0 ,lambda t: 0.99, lambda t: 0.005, lambda t: 0.005)
    G = nx.from_numpy_matrix(A)  # takes most of the time in this stage (~ 8 Sec)
    print("\tGragh load, Done!")
    # ----- creating other graphs ------
    # G = nx.barabasi_albert_graph(7000, m=3)
    dataset_name = 'TGPA'
    # ----- clustering coef -----
    cluster_coeff(G)
    # ----- p(k) and degree sequence -----
    plot_degree_dist(G, dataset_name)
    # ----- spectra of eigen values plot -----
    t.append(perf_counter())
    plot_spectra_eigen_vals(A, dataset_name)
    t.append(perf_counter())
    # ----- partial graph plot -----
    t.append(perf_counter())
    plot_graph(A, dataset_name)
    t.append(perf_counter())
    # ----
    # ----- profiling the code
    for i in range(1, len(t)):
        print("section {} took: {:.3f} sec".format(i, t[i] - t[i - 1]))


def cluster_coeff(G):
    low_cc_global = nx.average_clustering(G)  # runs in ~ 25 sec
    print("lower global clustering coeff:", low_cc_global)


def plot_degree_dist(G, dataset_name='Princeton12'):
    degree_sequence = list(np.array(G.degree())[:, 1])
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    cnt = np.array(cnt)
    cnt = cnt / sum(cnt)  # convert it into probability
    plt.figure()
    plt.loglog(deg, cnt, '.', markersize=6)
    plt.xlabel("log(Degree)");
    plt.ylabel("log(P(Degree))");
    plt.title("{} Degree distribution".format(dataset_name))
    plt.savefig("{}_1degree_dist_plot.png".format(dataset_name))


def plot_spectra_eigen_vals(A, dataset_name='Princeton12'):
    # lambda0 = nx.adjacency_spectrum(G)
    e = np.linalg.eigvals(A)
    # e2 = laplacian_spectrum(G)
    # TODO: loglog scale plot/hist (is spectra = Hist?!)
    plt.figure()
    plt.hist(e, bins=1000, rwidth=30)  # histogram with 100 bins
    plt.ylim(0,30)
    plt.xlim(0, np.max(e))  # eigenvalues between 0 and 2
    plt.ylabel("Values");
    plt.title("{} Adjacency matrix EigenValues".format(dataset_name))
    plt.savefig("{}_1eigen_spectra.png".format(dataset_name))
    # L = nx.normalized_laplacian_matrix(G)


def plot_graph(A, dataset_name='Princeton12'):
    '''
    bipartite_layout(G, nodes[, align, scale, …])  =  Position nodes in two straight lines.
    circular_layout(G[, scale, center, dim])  =  Position nodes on a circle.
    kamada_kawai_layout(G[, dist, pos, weight, …])  =  Position nodes using Kamada-Kawai path-length cost-function.
    planar_layout(G[, scale, center, dim])  =  Position nodes without edge intersections.
    random_layout(G[, center, dim, seed])  =   Position nodes uniformly at random in the unit square.
    rescale_layout(pos[, scale])  =  Returns scaled position array to (-scale, scale) in all axes.
    shell_layout(G[, nlist, scale, center, dim])  =  Position nodes in concentric circles.
    spring_layout(G[, k, pos, fixed, …])  =  Position nodes using Fruchterman-Reingold force-directed algorithm.
    spectral_layout(G[, weight, scale, center, dim])  =  Position nodes using the eigenvectors of the graph Laplacian.
    spiral_layout(G[, scale, center, dim, …])  =  Position nodes in a spiral layout.
    '''
    G = nx.from_numpy_matrix(A[0:200, 0:200])
    plt.figure()
    options = {
        "node_color": "blue",
        "node_size": 10,
        "line_color": "grey",
        "linewidths": 0.1,
        "width": 0.1,
    }
    nx.draw(G, pos=nx.spring_layout(G), **options)  # todo: try other style to get better visuals (like the paper)
    plt.title("{} Graph plot".format(dataset_name))
    plt.margins(-0.25)
    plt.savefig("{}_1graph_nodes_plot.png".format(dataset_name))


if __name__ == "__main__":
    main()