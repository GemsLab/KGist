from graph import Graph
from model import Model
from evaluator import Evaluator
import argparse
import numpy as np
import itertools
import functools
import random
from math import log2 as log
import heapq
from multiprocessing import Pool

class Searcher:
    '''
    A class which searches for the optimal model.
    '''
    def __init__(self, graph):
        self.graph = graph
        self.evaluator = Evaluator(graph)
        self.candidates = list(self.graph.candidates.keys())

    def rank_rules(self):
        '''
        Rank the candidate rules - (Section 4.1.3)
        '''
        # L(G|M_0)
        null_val = self.evaluator.length_graph_with_model(Model(self.graph))[0]
        def reduction_in_error(g):
            '''
            :g: a rule
            '''
            _model = Model(self.graph)
            _model.add_rule(g)
            # compute L(G|M_0) - L(G|M \cup {g})
            red_in_err = null_val - self.evaluator.length_graph_with_model(_model)[0]
            return red_in_err
        self.candidates = sorted(self.candidates,
                                 reverse=True,
                                 key=lambda g: (reduction_in_error(g), # reduction in error
                                                len(self.graph.candidates[g]['ca_to_size']), # number of correct assertions
                                                g[0])) # rule root labels

    class BoundedMinHeap:
        '''
        A Min Heap that only allows :bound: items.
        If :bound: is << len(items), then this can be more efficient for finding the top k than sorting the whole list.
        '''
        def __init__(self, bound, key=lambda it: it):
            self.bound = bound
            self.key = key
            self._data = list()

        def push(self, it):
            if len(self._data) < self.bound:
                heapq.heappush(self._data, (self.key(it), it))
            else:
                heapq.heappushpop(self._data, (self.key(it), it))

        def get_reversed(self):
            temp = list()
            while len(self._data) > 0:
                temp.append(heapq.heappop(self._data)[1])
            return list(reversed(temp))

    def build_model_top_k_freq(self, k):
        '''
        Build a model containing the k rules with the most correct assertions.
        '''
        model = Model(self.graph)
        heap = Searcher.BoundedMinHeap(bound=k, key=lambda rule: len(self.graph.candidates[rule]['ca_to_size']))
        for rule in self.candidates:
            heap.push(rule)
        for rule in heap.get_reversed():
            model.add_rule(rule)
        return model

    def build_model_top_k_coverage(self, k):
        '''
        Build a model containing the k rules that explain the most edges.
        '''
        model = Model(self.graph)
        heap = Searcher.BoundedMinHeap(bound=k, key=lambda rule: sum(list((self.graph.candidates[rule]['ca_to_size']).values())))
        for rule in self.candidates:
            heap.push(rule)
        for rule in heap.get_reversed():
            model.add_rule(rule)
        return model

    def check_qualify(self, evaluator, verbosity, rule_and_new_labels):
        '''
        Check whether adding more labels to a rule (qualifying it) leads to improvements in MDL terms.

        :evaluator: an Evaluator object with which to compute MDL scores
        :verbosity: how much to print to the log
        :rule_and_new_labels: (rule, the rule's new labels if qualified)

        :return: True if the rule is qualified, False if not.
        '''
        rule, new_labels = rule_and_new_labels
        old_rule = rule
        root, children = rule
        rule = (new_labels, children)
        # create M_0
        model = Model(self.graph)
        # create M_0 \cup {g without qualification}
        model.add_rule(old_rule)
        # compute L(G, M_0 \cup {g without qualification})
        # (Note: this computation works because the newly added labels are contained by all correct assertion starting points, so L(G|M) does not change)
        cost_without_qualification = evaluator.length_rule(old_rule) + evaluator.length_rule_assertions(old_rule, model)
        # replace the rule with the qualified version to make M_0 \cup {g with qualification}
        model.rules[rule] = model.rules[old_rule]
        del model.rules[old_rule]
        # compute L(G, M_0 \cup {g with qualification})
        cost_with_qualification = evaluator.length_rule(rule) + evaluator.length_rule_assertions(rule, model)

        qualified = False
        # if the cost went down, then keep the qualifiers
        if cost_with_qualification < cost_without_qualification:
            # update the data structures so that the new, qualified rules has the same correct assertions as the old, unqualified rule
            self.graph.candidates[rule] = self.graph.candidates[old_rule]
            # delete the data on the old, unqualified rule
            del self.graph.candidates[old_rule]
            return True
        return False

    def label_qualify(self, verbosity):
        '''
        Qualify rules where appropriate (Section 4.1.1).

        :verbosity: How often to print progress.
        '''
        num_qualified = 0
        rule_to_new_labels = dict()
        if verbosity > 0:
            print('Qualifying candidate rules (Section 4.1.2).')
        n = len(self.candidates)
        for i, rule in enumerate(self.candidates):
            root = rule[0][0]
            heads = list(self.graph.candidates[rule]['ca_to_size'].keys())
            # if all correct assertions share more labels than the current root label, then we can add this label
            shared_by_all = set(self.graph.labels(heads[0]))
            for head in heads[1:]:
                shared_by_all = shared_by_all.intersection(self.graph.labels(head))
                if shared_by_all == {root}:
                    break
            if shared_by_all != {root}:
                rule_to_new_labels[rule] = tuple(sorted(shared_by_all))
            if verbosity > 0 and i > 0 and i % verbosity == 0:
                print('{}% of candidates processed.'.format(round(i / n * 100, 2)))
        if verbosity > 0:
            print('{}% of candidates processed.'.format(round(i / n * 100, 2)))

        evaluator = Evaluator(self.graph)

        n = len(rule_to_new_labels.items())
        for i, rule_and_new_labels in enumerate(rule_to_new_labels.items()):
            # check whether the qualifier leads to improvements
            qualified = self.check_qualify(evaluator, verbosity, rule_and_new_labels)
            if qualified:
                num_qualified += 1
            if verbosity > 0 and i > 0 and i % verbosity == 0:
                print('{}'.format(i / n))

        self.candidates = list(self.graph.candidates.keys())
        if verbosity > 0:
            print('{}% of candidates qualified.'.format(round(num_qualified / n * 100, 2)))

    def build_model(self, rank='metric', order=['mdl_err', 'coverage', 'lex'], passes=2, label_qualify=True, verbosity=1000000):
        '''
        The core of the algorithm. (Sections 4.1.2-4.2.1)
        '''
        if label_qualify:
            self.label_qualify(verbosity=verbosity)

        if rank == 'metric':
            self.rank_rules()
        elif rank == 'random':
            print('Random order.')
            random.shuffle(self.candidates)
        # starts null
        model = Model(self.graph)

        best_val, best_model_length, best_neg_edge, best_neg_node = self.evaluator.evaluate(model, with_lengths=True)
        null_cost = best_val
        if verbosity > 0:
            print('Null encoding cost: {}'.format(round(best_val, 4)))
        num_cans = len(self.candidates)
        tried = set()

        for _pass in range(1, passes + 1):
            if verbosity > 0:
                print('Starting pass {}.'.format(_pass))
            for i, rule in enumerate(self.candidates):
                # build reverse
                root, child = rule[0], rule[1][0]
                reverse_rule = (child[2][0], ((child[0], 'in' if child[1] == 'out' else 'out', (root, ())),))
                if reverse_rule in self.graph.candidates:
                    acc_rule = len(self.graph.candidates[rule]['ca_to_size']) / self.graph.nodes_with_type(rule[0])
                    acc_rev_rule = len(self.graph.candidates[reverse_rule]['ca_to_size']) / self.graph.nodes_with_type(reverse_rule[0])

                    if rule in model.rules or reverse_rule in model.rules:
                        continue

                    model.add_rule(rule)
                    val, model_length, neg_edge, neg_node = self.evaluator.evaluate_change(model, rule, best_model_length)
                    model.remove_rule(rule)
                    model.add_rule(reverse_rule)
                    rev_val, rev_model_length, rev_neg_edge, rev_neg_node = self.evaluator.evaluate_change(model, reverse_rule, best_model_length)
                    model.remove_rule(reverse_rule)
                    # if the cost didn't go down, remove the rule
                    if val <= rev_val < best_val:
                        assert(val <= rev_val)
                        assert(val < best_val)
                        model.add_rule(rule)
                        best_val = val
                        best_model_length = model_length
                    elif rev_val < best_val:
                        assert(rev_val < best_val)
                        assert(rev_val < val)
                        model.add_rule(reverse_rule)
                        best_val = rev_val
                        best_model_length = rev_model_length
                else:
                    acc_rule = len(self.graph.candidates[rule]['ca_to_size']) / self.graph.nodes_with_type(rule[0])

                    if rule in model.rules:
                        continue

                    model.add_rule(rule)
                    val, model_length, neg_edge, neg_node = self.evaluator.evaluate_change(model, rule, best_model_length)

                    if val < best_val:
                        best_val = val
                        best_model_length = model_length
                    else: # if the cost didn't go down, remove the rule
                        model.remove_rule(rule)

                if verbosity > 0 and i > 0 and i % verbosity == 0:
                    print('Percent tried: {}. Num rules: {}. New encoding cost: {}. Percent saved: {}.'.format((i / num_cans) * 100, len(model.rules), best_val, ((null_cost - best_val) / null_cost) * 100))
            if verbosity > 0:
                print('Final number of rules after pass {}: {}'.format(_pass, len(model.rules)))

        # M*
        return model

if __name__ == '__main__':
    '''
    Entry point for the program.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph',
                        '-g',
                        type=str,
                        required=True,
                        help='the name of the graph to summarize')
    args = parser.parse_args()
    graph = Graph(args.graph)
