from collections import defaultdict
from evaluator import Evaluator
from itertools import chain, combinations
import os
import sys
import json
import networkx as nx
from rule import Rule
from realized_rule import RealizedRule
from copy import copy, deepcopy
import _pickle as pickle
import random

class Model:
    '''
    A model consisting of rules which explain a knowledge graph.
    '''
    def __init__(self, graph):
        '''
        :graph: the knowledge graph being modeled
        '''
        self.graph = graph
        # rules stored mapping to matches
        self.rules = dict()
        self.node_label_counts = defaultdict(int)
        self.tensor = set()
        self.label_matrix = set()
        self.cache = {'last_updated_rule': None}
        self.rule_graph = None
        self.shared_root_rule_dependency_graph = None
        self.subject_to_rules = dict()

    def add_rule(self, rule, ghost=False):
        '''
        :rule: the rule to be added
        :ghost: (**ONLY MEANT FOR TESTING**) parameter that can be set to true during testing to not add the assertions of a rule.
        '''
        # store rule
        if rule in self.rules:
            print('Already added')
            return

        if ghost:
            self.rules[rule] = 'ghost'
            return

        if type(rule) is not tuple:
            self.rules[rule] = 'level1'
            if rule.root not in self.subject_to_rules:
                self.subject_to_rules[rule.root] = set()
            self.subject_to_rules[rule.root].add(rule)
        else:
            self.rules[rule] = list(self.graph.candidates[rule]['ca_to_size'].values())
            if rule[0] not in self.subject_to_rules:
                self.subject_to_rules[rule[0]] = set()
            self.subject_to_rules[rule[0]].add(rule)
        self.make_assertions(rule)

    def remove_rule(self, rule):
        '''
        :rule: the rule to be removed
        '''
        if rule != self.cache['last_updated_rule']:
            print('We can only remove the last added rule.')
            return
        if rule not in self.rules: # make sure the rule is actually there
            return
        # remove rule
        del self.rules[rule]
        if type(rule) is not tuple:
            self.subject_to_rules[rule.root].remove(rule)
            if len(self.subject_to_rules[rule.root]) == 0:
                del self.subject_to_rules[rule.root]
        else:
            self.subject_to_rules[rule[0]].remove(rule)
            if len(self.subject_to_rules[rule[0]]) == 0:
                del self.subject_to_rules[rule[0]]

        self.undo_assertions(rule)

    def make_assertions(self, rule):
        '''
        Fills in model's tensor and node label map with assertions of a rule

        :rule: a rule
        '''
        # reset cache
        self.cache = {'last_updated_rule': rule}

        # update cache
        if type(rule) is not tuple:
            self.cache['new_edges'] = rule.get_edges_covered().difference(self.tensor)
            self.cache['new_labels'] = rule.get_labels_covered().difference(self.label_matrix)
        else:
            self.cache['new_edges'] = self.graph.candidates[rule]['edges'].difference(self.tensor)
            self.cache['new_labels'] = self.graph.candidates[rule]['label_coverage'].difference(self.label_matrix)

        self.tensor.update(self.cache['new_edges'])
        self.label_matrix.update(self.cache['new_labels'])

    def undo_assertions(self, rule):
        '''
        Removes things from model's tensor and node label map

        :rule: a rule
        '''
        self.tensor.difference_update(self.cache['new_edges'])
        self.label_matrix.difference_update(self.cache['new_labels'])

    def build_rule_graph(self, build_prime=False, force=False):
        '''
        Builds a dependency graph.
            - Nodes are rules
            - (r1, r2) means that r1's tail is r2's head and denotes a possible composition

        :return: a dependency graph of rules connected by their possible compositions
        '''
        if not force and self.rule_graph:
            return

        hub_to_rule = defaultdict(list)
        # find dependencies and store as edges for a dependency graph
        edges = list()
        tree_rules = list()
        for rule in self.rules:
            if type(rule) is tuple:
                parent, children = rule
                rule = Rule(parent, children)
            self.plant_forest(rule)
            hub_to_rule[rule.root].append(rule)
            tree_rules.append(rule)

        if not build_prime:
            # build graph
            for rule in tree_rules:
                # if leaf matches other rules' hubs
                if len(set(rule.get_leaves()).intersection(set(hub_to_rule.keys()))) > 0:
                    # get the matching rules
                    matching_rules = set()
                    for leaf in rule.get_leaves():
                        if leaf in hub_to_rule:
                            matching_rules.update(hub_to_rule[leaf])
                    for other_rule in matching_rules:
                        if rule.root in other_rule.get_leaves(): # don't allow loops
                            continue
                        edges.append((rule, other_rule))

            self.rule_graph = nx.DiGraph(edges)

        if build_prime:
            def jaccard_sim(r1, r2):
                a = set(real.root for real in r1.realizations)
                b = set(real.root for real in r2.realizations)
                return len(a.intersection(b)) / len(a.union(b)) if len(a.union(b)) > 0 else 0
            # build rule graph prime, which encodes dependencies of shared root types
            edges = list()
            for rule in tree_rules:
                for other_rule in hub_to_rule[rule.root]:
                    if rule != other_rule: #(other_rule, rule) not in edges:
                        self.plant_forest(rule)
                        self.plant_forest(other_rule)
                        if jaccard_sim(rule, other_rule) == 1.0:
                            edges.append((rule, other_rule))
            self.shared_root_rule_dependency_graph = nx.Graph(edges)

    def pickle_copy(self, obj):
        name = '{}'.format(random.randint(0, 10000000))
        with open('{}.pickle'.format(name), 'wb') as handle:
            pickle.dump(obj, handle)

        with open('{}.pickle'.format(name), 'rb') as handle:
            new_obj = pickle.load(handle)

        os.remove('{}.pickle'.format(name))

        return new_obj

    def merge_rules(self, verbosity):
        '''
        Perfrom merging refinement (Rm).

        :return: A new Model object with Rm applied to self.
        '''
        self.build_rule_graph(build_prime=True)
        merged = set()
        merged_model = Model(self.graph)
        # find sets of rules that all have the same root (i.e., cliques in the dependency graph)
        cliques = list(nx.find_cliques(self.shared_root_rule_dependency_graph))
        n = len(cliques)
        for i, clique in enumerate(cliques):
            # create a new rule composed that will consist of those in the clique merged
            new_rule = self.pickle_copy(clique[0])
            for r2 in clique[1:]:
                new_rule.merge(r2)
            merged_model.add_rule(new_rule)
            if verbosity > 0 and i > 0 and i % 100 == 0:
                print('{}% of candidate merges tested.'.format(round(i / n * 100, 2)))
        if verbosity > 0:
            print('{}% of candidate merges tested.'.format(round(i / n * 100, 2)))

        covered = set(node.tuplify() for node in self.shared_root_rule_dependency_graph.nodes())
        for rule in self.rules.keys():
            if rule not in covered:
                merged_model.add_rule(rule)

        return merged_model

    def plant_forest(self, rule):
        '''
        Grow a forest of realizations for a level-0 rule
        '''
        if rule.realized():
            return
        leaves = list()
        roots = set(self.graph.candidates[rule.tuplify()]['ca_to_size'].keys())
        # maps hubs to the realizations where they are the root
        hubs_to_realizations = dict()
        # iterate over edges explained by rule and construct stars
        for eid in self.graph.candidates[rule.tuplify()]['edges']:
            sub, pred, obj = self.graph.id_to_edge[eid]
            dir = rule.children[0][1]
            # decide which is hub and spoke
            hub = sub if dir == 'out' else obj
            spoke = obj if dir == 'out' else sub
            u, v = hub, spoke
            # create a rule realization for the hub
            if hub not in hubs_to_realizations:
                hubs_to_realizations[hub] = RealizedRule(hub, label=rule.root)
            hub = hubs_to_realizations[hub]
            # add edge (u, u_typ, pred, dir, v, v_typ)
            edge = (u, rule.root, pred, dir, v, rule.children[0][2].root)
            hub.add_edge(edge, eid=eid, labels=True)
        # iterate over each star (rule realization), and add to the rule (at the root level)
        for hub, realization in hubs_to_realizations.items():
            rule.insert_realization(realization)

    def nest_rules(self, verbosity):
        '''
        Perfrom nesting refinement (Rn).

        :return: A new Model object with Rn applied to self.
        '''
        def powerset(iterable):
            s = list(iterable)
            return set(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))

        self.build_rule_graph()

        def compose(r1, r2):
            '''
            :r1: a Rule
            :r2: a Rule
            '''
            composed_rule = self.pickle_copy(r1)
            assert(composed_rule.pin_to_leaf(r2))
            composed_rule.filter_errant()
            return composed_rule

        evaluator = Evaluator(self.graph)
        null_val = evaluator.evaluate(Model(self.graph)) # for log
        best_model = self
        last_model = None
        best_val = evaluator.evaluate(best_model)
        _graph = nx.DiGraph(self.rule_graph.edges())
        seen_pairs = set()
        def checkable(r):
            return r if type(r) is tuple else r.tuplify()

        edge_to_jaccard_sim = dict()
        for edge in _graph.edges():
            edge_to_jaccard_sim[edge] = edge[0].jaccard_sim(edge[1])

        num_nested = 0
        while last_model != best_model:
            num_rules = len(best_model.rules)
            rules = set(best_model.rules.keys())
            last_model = best_model
            pairs = _graph.edges()
            pairs = sorted(pairs, key=lambda edge: [edge[0].tuplify()] + [edge[1].tuplify()])
            pairs = sorted(pairs, reverse=True, key=lambda edge: edge_to_jaccard_sim[edge])
            pair_num = 0
            for r1, r2 in pairs:
                pair_num += 1
                if (r1, r2) in seen_pairs or checkable(r1) == checkable(r2):
                    continue
                seen_pairs.add((r1, r2))
                # pair is two rules that are candidates for composition
                model_to_test = Model(self.graph)
                candidate = compose(r1, r2)
                if len(candidate.realizations) == 0:
                    continue
                approx_L = best_val - evaluator.length_rule(r1) - evaluator.length_rule_assertions(r1, best_model) - evaluator.length_rule(r2) - evaluator.length_rule_assertions(r2, best_model) + evaluator.length_rule(candidate) + evaluator.length_rule_assertions(candidate, best_model)
                if approx_L >= best_val:
                    continue
                for rule in rules:
                    if checkable(rule) != checkable(r1) and checkable(rule) != checkable(r2):
                        model_to_test.add_rule(rule)

                model_to_test.add_rule(candidate)
                # evaluate the possible new model
                new_val = evaluator.evaluate(model_to_test)

                if new_val < best_val:
                    num_nested += 1
                    best_model = model_to_test
                    best_val = new_val
                    before_n = _graph.number_of_nodes()
                    before_m = _graph.number_of_edges()
                    _graph = nx.algorithms.minors.contracted_nodes(_graph, r1, r2, self_loops=False)
                    assert(_graph.number_of_nodes() == before_n - 1)
                    _graph = nx.relabel_nodes(_graph, {r1: candidate})
                    for edge in list(_graph.out_edges(candidate)) + list(_graph.in_edges(candidate)):
                        if edge not in edge_to_jaccard_sim:
                            edge_to_jaccard_sim[edge] = edge[0].jaccard_sim(edge[1])

                    to_remove = list()
                    # not all are legal
                    for comp, y in _graph.out_edges(candidate): # edges where comp's inner match y's root
                        assert(comp == candidate)
                        if y.root not in comp.get_inner_nodes():
                            to_remove.append((comp, y))
                    for y, comp in _graph.in_edges(candidate): # edges where y's inner match comp's root
                        assert(comp == candidate)
                        if comp.root not in y.get_inner_nodes():
                            to_remove.append((y, comp))
                    _graph.remove_edges_from(to_remove)

                    assert(_graph.number_of_nodes() == before_n - 1)
                    break
                if verbosity > 0 and pair_num % 100 == 0:
                    print('{}% of candidate compositions processed.'.format(round((pair_num / len(pairs)) * 100, 2)))
        if verbosity > 0:
            print('{}% of candidate compositions processed.'.format(round((pair_num / len(pairs)) * 100, 2)))
            print('{} nestings performed.'.format(num_nested))
        return best_model

    def build_from_model(self, model):
        for rule in model.rules:
            if type(rule) is tuple:
                self.add_rule(rule)
            else:
                for atom in rule.get_atoms():
                    self.add_rule(atom)

    def save(self, path='../data/output/model.json'):
        '''
        Saves the model in json format.

        :path: path to a json file where model should be saved
        '''
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        outfile = os.path.join(ROOT_DIR, path)

        data = dict()
        data['rules'] = list()
        for rule in self.rules:
            cas = defaultdict(set)
            for eid in self.graph.candidates[rule]['edges']:
                sub, pred, obj = self.graph.id_to_edge[eid]
                if self.graph.idify:
                    hub = sub if rule[1][0][1] == 'out' else obj
                    spoke = obj if rule[1][0][1] == 'out' else sub
                    cas[hub].add(spoke)
            correct_hubs = set(cas.keys())
            exceptions = list(self.graph.nodes_with_type(rule[0], num_only=False).difference(correct_hubs))
            if self.graph.idify:
                hub_labels = list(self.graph.id_to_label[label] for label in rule[0])
                spoke_labels = list(self.graph.id_to_label[label] for label in rule[1][0][2][0])
                edge_type = self.graph.id_to_pred[rule[1][0][0]]
                exceptions = list(self.graph.id_to_node[node] for node in exceptions)
            data['rules'].append({'rule': {'hub_labels': hub_labels,
                                           'edge_type': edge_type,
                                           'spoke_labels': spoke_labels,
                                           'dir': rule[1][0][1]},
                                  'correct_assertions': list(),
                                  'exceptions': exceptions})
            for hub, spokes in cas.items():
                if self.graph.idify:
                    hub = self.graph.id_to_node[hub]
                    spokes = list(self.graph.id_to_node[spoke] for spoke in spokes)
                data['rules'][-1]['correct_assertions'].append({'hub': hub, 'spokes': list(spokes)})

        with open(outfile, 'w') as f:
            json.dump(data, f, indent=4)

    def percent_improved(self):
        evaluator = Evaluator(self.graph)
        null_val = evaluator.evaluate(Model(self.graph))
        val = evaluator.evaluate(self)
        # the difference is what percent of the original?
        return ((null_val - val) / null_val) * 100

    def print_stats(self):
        evaluator = Evaluator(self.graph)
        val = evaluator.evaluate(self)
        print('----- Model stats -----')
        print('L(G,M) = {}'.format(round(val, 2)))
        null_val = evaluator.evaluate(Model(self.graph))
        print('% Bits needed: {}'.format(round((val / null_val) * 100, 2)))
        print('# Rules: {}'.format(len(self.rules)))
        print('% Edges Explained: {}'.format(round(len(self.tensor) / self.graph.m * 100, 2)))
        print('-----------------------')
