from math import log2 as log
from scipy.special import gammaln
from collections import defaultdict
import numpy as np
from itertools import chain, combinations
from evaluator import Evaluator

class AnomalyDetector:
    def __init__(self, model):
        self.idify = model.graph.idify
        self.model = model
        self.evaluator = Evaluator(self.model.graph)
        self.subject_to_rules = dict()
        for rule in self.model.rules:
            sub = rule[0] if type(rule) is tuple else rule.root
            if sub not in self.subject_to_rules:
                self.subject_to_rules[sub] = set()
            self.subject_to_rules[sub].add(rule)

        self.binomial_map = dict()
        self.edge_to_id = dict()
        for m, edge in model.graph.id_to_edge.items():
            sub, pred, obj = edge
            self.edge_to_id[(sub, pred, obj)] = m

    def length_binomial(self, n, k):
        '''
        :n: n in (n choose k)
        :k: k in (n choose k)

        :return: log_2(n choose k)
        '''
        if (n, k) in self.binomial_map:
            return self.binomial_map[(n, k)]

        length = (gammaln(n + 1) - gammaln(k + 1) - gammaln((n + 1) - k)) / np.log(2)
        self.binomial_map[(n, k)] = length
        return length

    def score_blame_edge(self, node, pred):
        '''
        The number of bits describing the node as an exception to rules.
        '''
        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

        score = 0
        rules = set()
        for labels in powerset(self.model.graph.labels(node)):
            if labels in self.subject_to_rules:
                rules.update(self.subject_to_rules[labels])

        def has_pred(rule, pred):
            if type(rule) is tuple:
                return rule[1][0][0] == pred
            else:
                return pred in rule.get_preds()

        rules = set(filter(lambda rule: has_pred(rule, pred), rules))

        for rule in rules:
            cas = set(self.model.graph.candidates[rule]['ca_to_size'].keys()) if type(rule) is tuple else set(real.root for real in rule.correct_assertions)
            if node not in cas:
                num_assertions = self.model.graph.nodes_with_type(rule[0] if type(rule) is tuple else rule.root)
                num_correct = len(self.model.graph.candidates[rule]['ca_to_size'] if type(rule) is tuple else rule.correct_assertions)
                num_exceptions = num_assertions - num_correct
                assert(num_assertions > 0)
                assert(num_exceptions > 0)
                score += (1 / num_exceptions) * self.length_binomial(num_assertions, num_exceptions)
        return score

    def score_edge(self, edge, blame_edge=True):
        if self.idify:
            edge = (self.model.graph.node_to_id[edge[0]], self.model.graph.pred_to_id[edge[1]], self.model.graph.node_to_id[edge[2]])
        eid = self.edge_to_id[edge] if type(edge) is tuple else edge
        sub, pred, obj = edge if type(edge) is tuple else self.model.graph.id_to_edge[eid]
        score = 0
        if eid not in self.model.tensor: # edges not in the model are not explained
            # number of bits describing the unexplained edge
            score = (1 / (self.model.graph.m - len(self.model.tensor))) * self.evaluator.length_negative_edge_error(self.model)
        score += self.score_blame_edge(sub, pred) + self.score_blame_edge(obj, pred)
        return score
