import torch
import numpy as np
import torch.nn as nn
import pdb
import math
import copy


class Graph:
    """The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self, layout='custom', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # 'body', 'left', 'right', 'mouth', 'face'
        # if layout == 'custom_hand21':

        if layout == 'default_left' or layout == 'default_right':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [0, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [0, 9],
                [9, 10],
                [10, 11],
                [11, 12],
                [0, 13],
                [13, 14],
                [14, 15],
                [15, 16],
                [0, 17],
                [17, 18],
                [18, 19],
                [19, 20],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'default_body':
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [3, 5],
                [5, 7],
                [4, 6],
                [6, 8],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0

        elif layout == 'default_face_all':
            self.num_node = 9 + 8 + 1
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[i, i + 1] for i in range(9 - 1)] + \
                             [[i, i + 1] for i in range(9, 9 + 8 - 1)] + \
                             [[9 + 8 - 1, 9]] + \
                             [[17, i] for i in range(17)]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = self.num_node - 1

        elif layout in ['default_ytasl_left', 'default_ytasl_right', 'pruned_ytasl_left', 'pruned_ytasl_right', 'isharah_ytasl_left', 'isharah_ytasl_right']:
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [3, 4],
                [0, 5],
                [17, 18],
                [0, 17],
                [13, 14],
                [13, 17],
                [18, 19],
                [5, 6],
                [5, 9],
                [14, 15],
                [0, 1],
                [9, 10],
                [1, 2],
                [9, 13],
                [10, 11],
                [19, 20],
                [6, 7],
                [15, 16],
                [2, 3],
                [11, 12],
                [7, 8]
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0

        elif layout == 'default_ytasl_body':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [15, 21],
                [16, 20],
                [18, 20],
                [3, 7],
                [14, 16],
                [11, 23],
                [6, 8],
                [15, 17],
                [16, 22],
                [4, 5],
                [5, 6],
                [12, 24],
                [23, 24],
                [0, 1],
                [9, 10],
                [1, 2],
                [0, 4],
                [11, 13],
                [15, 19],
                [16, 18],
                [12, 14],
                [17, 19],
                [2, 3],
                [11, 12],
                [13, 15]
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0

        elif layout == 'default_ytasl_face_all':
            self.num_node = 37
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [16, 18],
                [18, 13],
                [13, 7],
                [7, 8],
                [8, 15],
                [15, 24],
                [24, 23],
                [23, 29],
                [29, 32],
                [32, 16],
                [5, 17],
                [17, 14],
                [30, 31],
                [31, 21],
                [11, 1],
                [1, 27],
                [10, 6],
                [6, 0],
                [0, 22],
                [22, 26],
                [26, 34],
                [34, 4],
                [4, 20],
                [20, 10],
                [10, 12],
                [12, 2],
                [2, 28],
                [28, 26],
                [10, 19],
                [19, 3],
                [3, 33],
                [33, 26]
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = self.num_node - 1

        elif layout in ['pruned_ytasl_body', 'isharah_ytasl_body']:
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [3, 5],
                [4, 6],
                [5, 7],
                [6, 8],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0

        elif layout == 'pruned_ytasl_face_all':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [5, 8],
                [8, 6],
                [6, 14],
                [14, 12],
                [3, 4],
                [4, 1],
                [1, 11],
                [11, 10],
                [3, 9],
                [9, 2],
                [2, 15],
                [15, 10],
                [7, 16],
                [13, 17],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = self.num_node - 1

        elif layout == 'isharah_ytasl_face_all':
            self.num_node = 19
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 2],
                [2, 3],
                [3, 4],
                [4, 10],
                [10, 5],
                [5, 8],
                [8, 7],
                [7, 9],
                [9, 6],
                [6, 1],
                [1, 15],
                [15, 18],
                [18, 16],
                [16, 17],
                [17, 14],
                [14, 13],
                [13, 12],
                [12, 11],
                [11, 0],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = self.num_node - 1

        else:
            raise NotImplementedError(f"Layout not implemented for: {layout}")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (
                                self.hop_dis[j, self.center]
                                == self.hop_dis[i, self.center]
                            ):
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif (
                                self.hop_dis[j, self.center]
                                > self.hop_dis[i, self.center]
                            ):
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD