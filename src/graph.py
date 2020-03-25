import numpy as np
from collections import defaultdict
import os
from itertools import chain, combinations
from scipy.sparse import lil_matrix
import itertools
import sys
import _pickle as pickle

class Graph:
    '''
    A Knowledge Graph representation which seeks to make the operations needed
    for this method as efficient as possible.
    '''
    def __init__(self, name, ext='txt', delimiter=' ', idify=False, verbose=True, load_candidates_from_disk=False):
        '''
        :name: the name of the graph to load.
        :ext: the file extension for the graph files.
        :delimiter: the delimiter for the graph files.
        :idify: if true, converts strings to ids
        :verbose: whether or not to print simple stats (e.g., num nodes)

        Assumes:
        1) ../data/{name}.{ext} edgelist.
        2) ../data/{name}_labels.{ext} node label mapping.

        e.g., ../data/dbpedia.txt and ../data/dbpedia_labels.txt
        '''
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        edgelist = os.path.join(ROOT_DIR, '../data/{}.{}'.format(name, ext))
        label_map = os.path.join(ROOT_DIR, '../data/{}_labels.{}'.format(name, ext))
        size_path = os.path.join(ROOT_DIR, '../data/{}_size.{}'.format(name, ext))
        self.name = name
        self.node_list = set()
        self.node_to_labels = dict()
        self.node_label_counts = defaultdict(int)
        self.edge_label_counts = defaultdict(int)
        self.idify = idify
        # load the graph
        self.load(edgelist, label_map, delimiter, verbose, load_candidates_from_disk)
        self.num_node_labels = len(self.label_matrix)
        if verbose:
            print('|V| = {}'.format(self.n))
            print('|E| = {}'.format(self.m))
            print('|L_V| = {}'.format(self.num_node_labels))
            print('|L_E| = {}'.format(self.num_edge_labels))

    def load(self, edgelist, label_map, delimiter, verbose, load_candidates_from_disk):
        '''
        Loads a knowledge graph.

        :edgelist: a path to the graph edgelist.
        :label_map: a path to the node label mapping.
        :delimiter: the delimiter used in the above files.
        '''
        self.label_matrix = dict()
        if self.idify:
            nid = 0
            eid = 0
            lid = 0
            self.id_to_pred = dict()
            self.pred_to_id = dict()
            self.id_to_node = dict()
            self.node_to_id = dict()
            self.label_to_id = dict()
            self.id_to_label = dict()
        # read node label mapping
        with open(label_map, 'r') as f:
            for line in f:
                line = line.strip().split(delimiter)
                node = line[0]
                if self.idify:
                    self.id_to_node[nid] = node
                    self.node_to_id[node] = nid
                    node = nid
                    nid += 1
                self.node_list.add(node)
                labels = tuple(line[1:])
                self.node_to_labels[node] = labels
                for label in labels:
                    if self.idify:
                        if label not in self.label_to_id:
                            self.label_to_id[label] = lid
                            self.id_to_label[lid] = label
                            lid += 1
                        label = self.label_to_id[label]
                    self.node_label_counts[label] += 1
                    if label not in self.label_matrix:
                        self.label_matrix[label] = set()
                    self.label_matrix[label].add(node)
                if self.idify:
                    self.node_to_labels[node] = tuple(self.label_to_id[label] for label in line[1:])
        if verbose:
            print('Node labels loaded.')

        # candidates map to matches
        self.candidates = dict()
        self.m = 0
        self.id_to_edge = dict()
        # read edgelist
        edge_labels = set()
        with open(edgelist, 'r') as f:
            self.tensor = set()
            for line in f:
                sub, pred, obj = line.strip().split(delimiter)
                edge_labels.add(pred)
                if self.idify:
                    if sub not in self.node_to_id:
                        self.node_to_id[sub] = nid
                        self.id_to_node[nid] = sub
                        nid += 1
                    sub = self.node_to_id[sub]
                    if obj not in self.node_to_id:
                        self.node_to_id[obj] = nid
                        self.id_to_node[nid] = obj
                        nid += 1
                    obj = self.node_to_id[obj]
                    if pred not in self.pred_to_id:
                        self.pred_to_id[pred] = eid
                        self.id_to_pred[eid] = pred
                        eid += 1
                    pred = self.pred_to_id[pred]
                self.edge_label_counts[pred] += 1
                self.node_list.add(sub)
                self.node_list.add(obj)
                self.tensor.add(self.m)
                if not load_candidates_from_disk:
                    # candidates
                    sub_labels = self.labels(sub)
                    # obj labels
                    obj_labels = self.labels(obj)
                    sls_ols = list(itertools.product(sub_labels, obj_labels))
                    for sl, ol in sls_ols:
                        if ((sl,), ((pred, 'out', ((ol,), ())),)) not in self.candidates:
                            self.candidates[((sl,), ((pred, 'out', ((ol,), ())),))] = {'label_coverage': set(),
                                                                                'edges': set(),
                                                                                'ca_to_size': defaultdict(int)}
                        self.candidates[((sl,), ((pred, 'out', ((ol,), ())),))]['label_coverage'].add((ol, obj))
                        self.candidates[((sl,), ((pred, 'out', ((ol,), ())),))]['edges'].add(self.m)
                        self.candidates[((sl,), ((pred, 'out', ((ol,), ())),))]['ca_to_size'][sub] += 1

                        if ((ol,), ((pred, 'in', ((sl,), ())),)) not in self.candidates:
                            self.candidates[((ol,), ((pred, 'in', ((sl,), ())),))] = {'label_coverage': set(),
                                                                               'edges': set(),
                                                                               'ca_to_size': defaultdict(int)}
                        self.candidates[((ol,), ((pred, 'in', ((sl,), ())),))]['label_coverage'].add((sl, sub))
                        self.candidates[((ol,), ((pred, 'in', ((sl,), ())),))]['edges'].add(self.m)
                        self.candidates[((ol,), ((pred, 'in', ((sl,), ())),))]['ca_to_size'][obj] += 1
                self.id_to_edge[self.m] = (sub, pred, obj)
                self.m += 1
                if verbose and self.m % 10000 == 0:
                    sys.stdout.write('\r{}'.format(self.m))
                    sys.stdout.flush()
        typ_to_nodes = dict()
        for candidate, info in self.candidates.items():
            if candidate[0] not in typ_to_nodes:
                typ_to_nodes[candidate[0]] = set((candidate[0][0], node) for node in self.nodes_with_type(candidate[0], num_only=False))
            # info['label_coverage'].update(typ_to_nodes[candidate[0]])
        self.node_list = list(self.node_list)
        self.num_edge_labels = len(edge_labels)
        self.total_num_labels = 0
        for label, nodes in self.label_matrix.items():
            self.total_num_labels += len(nodes)
        self.n = len(self.node_list)

        if load_candidates_from_disk:
            with open('../data/{}_candidates.pickle'.format(self.name), 'rb') as f:
                self.candidates = pickle.load(f)

        if verbose:
            print()

    def nodes(self):
        return self.node_list

    def labels(self, node):
        return self.node_to_labels[node]

    def nodes_with_type(self, typ, num_only=True):
        if len(typ) == 1:
            return len(self.label_matrix[typ[0]]) if num_only else self.label_matrix[typ[0]]
        return len(set.intersection(*list(self.label_matrix[label] for label in typ))) if num_only else set.intersection(*list(self.label_matrix[label] for label in typ))

    def tuplify(self, rule):
        if self.idify:
            return (tuple(self.id_to_label[label] for label in rule[0]), tuple((self.id_to_pred[child[0]], child[1], self.tuplify(child[2])) for child in rule[1]))
        return (rule[0], tuple((child[0], child[1], self.tuplify(child[2])) for child in rule[1]))
