from math import log2 as log
from scipy.special import comb
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
from scipy.special import gammaln
from rule import Rule

class Evaluator:
    '''
    Evaluates a model or a rule.
    '''
    def __init__(self, graph):
        self.graph = graph
        self.log_V = log(self.graph.n)
        self.log_E = log(self.graph.m)
        self.log_E_plus_1 = log(self.graph.m + 1)
        self.log_labels_plus_1 = log(self.graph.total_num_labels + 1)
        self.log_LV = log(self.graph.num_node_labels)
        # a cache for L_N since this can be slow to do many times
        self.length_natural_number_map = dict()
        # a cache for log of binomials
        self.binomial_map = dict()
        self.rule_to_length = dict()

    def evaluate(self, model, with_lengths=False):
        '''
        L(M) + L(G|M) - (Section 3.2)
        '''
        length_model = self.length_model(model)
        length_error, neg_edge, neg_node = self.length_graph_with_model(model)
        val = length_model + length_error
        if with_lengths:
            return val, length_model, neg_edge, neg_node
        return val

    def evaluate_change(self, model, rule, prev_model_len):
        '''
        L(M union {r}) + L(G|M union {r})

        :return: the MDL objective after inserting a new rule to M
        '''
        # update score
        neg_edge = self.length_negative_edge_error(model)
        neg_node = self.length_negative_label_error(model)
        length_error_with_rule = neg_edge + neg_node
        length_model_with_rule = self.length_model_new_rule(model, rule, prev_model_len)
        val = length_model_with_rule + length_error_with_rule

        return val, length_model_with_rule, neg_edge, neg_node

    def length_model_new_rule(self, model, rule, length_model):
        '''
        L(M union {r})

        :return: the length of the model with a new rule added
        '''
        # old length
        length = length_model
        length += self.length_rule(rule) + self.length_rule_assertions(rule, model)

        return length

    def length_natural_number(self, n):
        '''
        :n: a number

        :return: the number of bits required to transmit the number
        '''
        if n <= 0:
            return 0
        if n in self.length_natural_number_map:
            return self.length_natural_number_map[n]
        c = log(2.865064)
        i = log(n)
        while i > 0:
            c = c + i
            i = log(i)
        self.length_natural_number_map[n] = c # cache the value
        return c

    def length_binomial(self, n, k):
        '''
        :n: n in (n choose k)
        :k: k in (n choose k)

        :return: log_2(n choose k)
        '''
        # we cache computations that we've already made for speedups
        if (n, k) in self.binomial_map:
            return self.binomial_map[(n, k)]

        # gammln computation of log_e(n choose k) with change of base to log_2
        length = (gammaln(n + 1) - gammaln(k + 1) - gammaln((n + 1) - k)) / np.log(2)
        # cache the result for future use
        self.binomial_map[(n, k)] = length
        return length

    def length_model(self, model):
        '''
        L(M) - (Section 3.2.1)

        :model: a list of rules and their (partial-)matches

        :return: the number of bits required to transmit the model
        '''
        rules = model.rules.keys()
        # num rules
        length = log(2 * self.graph.num_node_labels * self.graph.num_edge_labels * self.graph.num_node_labels + 1)
        # rules
        length += sum(self.length_rule(rule) + self.length_rule_assertions(rule, model) for rule in rules)

        return length

    def length_rule(self, rule):
        '''
        L(g) - (Section 3.2.1)

        :rule: (parent, children)
            - recusrive case
                - parent: a set of node labels
                - children: a set of elements like (edge_type, dir, rule)
            - base case (parent is a leaf)
                - parent: a set of node labels
                - children: an empty set

        :return: the number of bits required to transmit a rule
        '''
        if type(rule) is not tuple:
            '''
            Level 1+ rules.
            '''
            # node labels
            length = self.log_LV
            length += sum(-log(self.graph.node_label_counts[label] / self.graph.n) for label in rule.root)
            # num branches
            length += self.length_natural_number(len(rule.children) + 1)
            for child in rule.children:
                edge_type, _, child = child
                # edge dir
                length += 1
                # edge label
                length += -log(self.graph.edge_label_counts[edge_type] / self.graph.m)
                # children
                length += self.length_rule(child)
            return length

        parent, children = rule

        # parent labels
        length = self.log_LV
        length += sum(-log(self.graph.node_label_counts[label] / self.graph.n) for label in parent)
        # num branches
        length += self.length_natural_number(len(children) + 1)
        for child in children:
            edge_type, _, rule_prime = child
            # edge type
            length += -log(self.graph.edge_label_counts[edge_type] / self.graph.m)
            # edge dir
            length += 1
            # visit child
            length += self.length_rule(rule_prime)
        return length

    def length_rule_assertions(self, rule, model, correct_assertions=None, info=False):
        '''
        L(alpha(g)) - (Section 3.2.1)
        '''
        if type(rule) is not tuple:
            '''
            Level 1+ rules.
            '''
            if rule.tuplify() in self.rule_to_length:
                return self.rule_to_length[rule.tuplify()]
            # num assertions
            num_assertions = self.graph.nodes_with_type(rule.root)
            # num exceptions
            num_correct = len(rule.realizations)
            num_exceptions = num_assertions - num_correct
            length = log(num_assertions)
            # exception ids
            length += self.length_binomial(num_assertions, num_exceptions)
            cost_of_exceptions = length
            # correct assertions
            def length_realization(real, real_node, _rule, depth=1):
                '''
                :real: a RealizedRule, which we descend recusrively to encode
                :_rule: a Rule, of which real is a relization
                '''
                _length = 0
                # for r' in chi(r)
                for pred, dir, child in _rule.children:
                    # num spokes
                    _length += self.log_V
                    # spoke ids
                    num_assertions = len(real.nodes[real_node].neighbors_of_type[(pred, dir, child.root)])
                    _length += self.length_binomial(self.graph.n - 1, num_assertions)
                    # visit children
                    for real_child in real.nodes[real_node].neighbors_of_type[(pred, dir, child.root)]:
                        _length += length_realization(real, real_child, child, depth + 1)
                return _length

            for real in rule.realizations: # each realization is a tree
                length += length_realization(real, real.root, rule)

            self.rule_to_length[rule.tuplify()] = length
            if info:
                # length, length of excpetions, length of cas
                return length, cost_of_exceptions, length - cost_of_exceptions
            return length

        if rule in self.rule_to_length:
            return self.rule_to_length[rule]
        # correct assertions
        correct_assertions = model.rules[rule] if model else correct_assertions
        # num assertions
        num_assertions = self.graph.nodes_with_type(rule[0])
        # num exceptions
        num_correct = len(correct_assertions)
        num_exceptions = num_assertions - num_correct
        length = log(num_assertions)
        # exception ids
        length += self.length_binomial(num_assertions, num_exceptions)
        cost_of_exceptions = length

        def length_rule_tree(rule, rho):
            # num_to_choose_from = self.graph.nodes_with_type(rule[1][0][2][0])
            # assert(num_to_choose_from >= rho)
            return self.log_V + self.length_binomial(self.graph.n - 1, rho)

        # correct assertions
        for ca in correct_assertions:
            # size correct assertion
            length += length_rule_tree(rule, ca)
        self.rule_to_length[rule] = length
        if info:
            return length, cost_of_exceptions, length - cost_of_exceptions
        return length

    def length_graph_with_model(self, model):
        '''
        L(G|M) - (Section 3.2.2)
        '''
        negative_edge_error = self.length_negative_edge_error(model)
        negative_node_error = self.length_negative_label_error(model)
        length = negative_edge_error + negative_node_error
        return length, negative_edge_error, negative_node_error

    def length_negative_edge_error(self, model):
        '''
        L(A-) - (Section 3.2.2)
        '''
        # number of ones modeled
        num_modeled = len(model.tensor)
        # num ones
        num_unexplained = self.graph.m - num_modeled
        # ones
        length = self.length_binomial((self.graph.n ** 2) * self.graph.num_edge_labels - num_modeled, num_unexplained)
        return length

    def length_negative_label_error(self, model):
        '''
        L(L-) - (Section 3.2.2)
        '''
        # number of ones modeled
        num_modeled = len(model.label_matrix)
        # num ones
        num_unexplained = self.graph.total_num_labels - num_modeled
        # ones
        length = self.length_binomial(self.graph.num_node_labels * self.graph.n - num_modeled, num_unexplained)
        return length
