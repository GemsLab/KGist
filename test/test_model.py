import unittest
import sys
sys.path.append('../src/')
from graph import Graph
from model import Model
from searcher import Searcher
from evaluator import Evaluator
from scipy.sparse import csr_matrix
import numpy as np
import mock
from copy import deepcopy
import networkx as nx
from rule import Rule

class TestModel(unittest.TestCase):

    def check(self, node, labels, graph, model):
        ca_labels = set(graph.node_to_labels[node])
        covered_labels = set(list(it[0] for it in filter(lambda ln: ln[1] == node, model.label_matrix)))
        assert(ca_labels.difference(covered_labels) == set(labels))

    def test_make_assertions_1(self):
        graph = Graph('test', verbose=False)
        rule = (('1927286',), (('6293378', 'out', (('7241965',), ())),))
        model = Model(graph)
        model.rules[rule] = list()
        model.make_assertions(rule)
        neg_edges = graph.m - len(model.tensor)
        assert(neg_edges == graph.m - 6)
        # everyone's labels should be explained
        self.check('36240', ('6843923',), graph, model)
        self.check('6175574', ('6843923',), graph, model)
        self.check('2415820', ('6843923',), graph, model)
        self.check('6341376', ('6843923',), graph, model)
        self.check('6555563', ('6843923',), graph, model)
        self.check('879961', ('6843923',), graph, model)

    def test_undo_assertions_1(self):
        graph = Graph('test', verbose=False)
        rule = (('1927286',), (('6293378', 'out', (('7241965',), ())),))
        model = Model(graph)
        model.rules[rule] = list()
        model.make_assertions(rule)
        model.undo_assertions(rule)
        # all explained edges are gone
        assert(len(model.tensor) == 0)
        neg_edges = 0
        neg_edges += len(graph.tensor.difference(model.tensor))
        assert(neg_edges == graph.m)
        # everyone's labels should be back to normal
        self.check('7499850', ('1927286',), graph, model)
        self.check('36240', ('7241965', '6843923'), graph, model)
        self.check('6175574', ('7241965', '6843923'), graph, model)
        self.check('2415820', ('7241965', '6843923'), graph, model)
        self.check('6341376', ('7241965', '6843923'), graph, model)
        self.check('6555563', ('7241965', '6843923'), graph, model)
        self.check('879961', ('7241965', '6843923'), graph, model)

    def test_make_assertions_2(self):
        graph = Graph('test', verbose=False)
        rule = (('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model = Model(graph)
        model.rules[rule] = list()
        model.make_assertions(rule)
        # 2 students have been 5794125ed an 5794125
        assert(len(model.tensor) == 2)
        # 2 edges should be explained
        neg_edges = 0
        neg_edges += len(graph.tensor.difference(model.tensor))
        assert(neg_edges == graph.m - 2)
        # labels should be explained
        self.check('308389', (), graph, model)

    def test_undo_assertions_2(self):
        graph = Graph('test', verbose=False)
        rule = (('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model = Model(graph)
        model.rules[rule] = list()
        model.make_assertions(rule)
        model.undo_assertions(rule)
        # every6057655 should be back to normal
        assert(len(model.tensor) == 0)
        neg_edges = 0
        neg_edges += len(graph.tensor.difference(model.tensor))
        assert(neg_edges == graph.m)
        self.check('6555563', ('7241965', '6843923'), graph, model)
        self.check('6341376', ('7241965', '6843923',), graph, model)
        self.check('308389', ('5794125',), graph, model)

    def test_make_assertions_3(self):
        graph = Graph('test', verbose=False)
        rule = (('8226812',), (('6291253', 'in', (('7241965',), ())),))
        model = Model(graph)
        model.rules[rule] = list()
        model.make_assertions(rule)
        # two phds living in it
        assert(len(model.tensor) == 2)
        # 2 edges should be explained
        neg_edges = 0
        neg_edges += len(graph.tensor.difference(model.tensor))
        assert(neg_edges == graph.m - 2)
        # everyone's labels should be explained
        self.check('6175574', ('6843923',), graph, model)
        self.check('2415820', ('6843923',), graph, model)

    def test_undo_assertions_3(self):
        graph = Graph('test', verbose=False)
        rule = (('8226812',), (('6291253', 'in', (('7241965',), ())),))
        model = Model(graph)
        model.rules[rule] = list()
        model.make_assertions(rule)
        model.undo_assertions(rule)
        # should be cleared
        assert(len(model.tensor) == 0)
        neg_edges = 0
        neg_edges += len(graph.tensor.difference(model.tensor))
        assert(neg_edges == graph.m)
        # everyone's labels should be explained
        self.check('6175574', ('7241965', '6843923',), graph, model)
        self.check('2415820', ('7241965', '6843923',), graph, model)
        self.check('7992351', ('8226812',), graph, model)

    def test_add_rules(self):
        graph = Graph('test', verbose=False)
        rule1 = (('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = (('7241965',), (('5835005', 'out', (('5794125',), ())),))
        rule3 = (('8226812',), (('6291253', 'in', (('6843923',), ())),))
        model = Model(graph)
        assert(len(model.rules) == 0)
        # rule 1
        model.add_rule(rule1)
        assert(len(model.rules) == 1)
        assert(len(model.rules[rule1]) == 1)
        assert(model.rules[rule1][0] == 6)
        # rule 2
        model.add_rule(rule2)
        assert(len(model.rules) == 2)
        assert(len(model.rules[rule2]) == 2)
        assert(model.rules[rule2][0] == 1)
        assert(model.rules[rule2][1] == 1)
        # rule 3
        model.add_rule(rule3)
        assert(len(model.rules) == 3)
        assert(len(model.rules[rule3]) == 1)
        assert(model.rules[rule3][0] == 2)

        # everyone's labels should be explained
        self.check('36240', ('6843923',), graph, model)
        self.check('6175574', (), graph, model)
        self.check('2415820', (), graph, model)
        self.check('6341376', ('6843923',), graph, model)
        self.check('6555563', ('6843923',), graph, model)
        self.check('879961', ('6843923',), graph, model)
        self.check('308389', (), graph, model)

    def test_repeated_add_removal_1(self):
        graph = Graph('test', verbose=False)
        searcher = Searcher(graph)
        searcher.rank_rules()
        # starts null
        model = Model(graph)
        null_val, null_model_length, null_neg_edge, null_neg_node = searcher.evaluator.evaluate(model, with_lengths=True)
        model_length, neg_edge, neg_node = null_model_length, null_neg_edge, null_neg_node
        for rule in searcher.candidates:
            model.add_rule(rule)
            val, model_length, neg_edge, neg_node = searcher.evaluator.evaluate_change(model, rule, null_model_length)
            assert(model_length >= 0)
            assert(neg_edge >= 0)
            assert(neg_node >= 0)
            assert(model_length > null_model_length)
            assert(model_length != float('Inf'))
            assert(neg_edge < null_neg_edge)
            assert(neg_node <= null_neg_node)
            model.remove_rule(rule)
            assert((null_val, null_model_length, null_neg_edge, null_neg_node) == searcher.evaluator.evaluate(model, with_lengths=True))

    def test_repeated_add_removal_2(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        # starts null
        model = Model(graph)
        rule1 = (('1927286',), (('6293378', 'out', (('6843923',), ())),))
        rule2 = (('7241965',), (('6293378', 'in', (('1927286',), ())),))

        model.add_rule(rule1)
        val_before = evaluator.evaluate(model)
        model.add_rule(rule2)
        model.remove_rule(rule2)
        val_after = evaluator.evaluate(model)
        assert(val_before == val_after)

    def test_idified(self):
        graph = Graph('test', idify=True, verbose=False)
        assert('1927286' in graph.label_to_id)
        assert('6293378' in graph.pred_to_id)
        assert('7241965' in graph.label_to_id)
        rule = ((graph.label_to_id['1927286'],), ((graph.pred_to_id['6293378'], 'out', ((graph.label_to_id['7241965'],), ())),))
        model = Model(graph)
        model.add_rule(rule)

    def test_save(self):
        # TODO
        pass

    def test_build_rule_graph_1(self):
        '''
        Tests that rule graph edge do capture dependencies (i.e. matching root & leaves)
        '''
        graph = Graph('test', idify=True, verbose=False)
        searcher = Searcher(graph)
        model = searcher.build_model_top_k_coverage(k=5)
        model.build_rule_graph()
        assert(len(model.rules) == 5)
        for edge in model.rule_graph.edges():
            assert(edge[0].get_leaves()[0] == edge[1].root)

    def test_build_rule_graph_2(self):
        '''
        Tests that rule graph is built properly with merged rules.
        '''
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('1927286',), (('412681', 'out', (('7490702',), ())),)))
        model.add_rule((('1927286',), (('3320538', 'out', (('8226812',), ())),)))
        model.add_rule((('1927286',), (('6291253', 'out', (('8226812',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        model.add_rule((('7490702',), (('412681', 'in', (('7241965',), ())),)))
        merged = model.merge_rules(verbosity=0)
        merged.build_rule_graph()
        assert(merged.rule_graph.number_of_nodes() == 3)
        assert(merged.rule_graph.number_of_edges() == 3)
        for r1, r2 in merged.rule_graph.edges():
            assert(r2.root in r1.get_leaves())
            a = set()
            for ca in r1.correct_assertions:
                a.update(set(ca.nodes.keys()).difference({ca.root}))
            b = set(ca.root for ca in r2.correct_assertions)
            if r1.root == ('1927286',) and (r2.root == ('7490702',) or r2.root == ('7241965',)):
                assert(0 < len(a.intersection(b)) / len(a.union(b)) < 1)

    @mock.patch('model.Model.plant_forest')
    def test_build_rule_graph_3(self, plant_forest):
        '''
        Tests that rule graph is built properly with merged rules.
        '''
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('agproduct',), (('madeFrom', 'out', (('animal',), ())),)), ghost=True)
        model.add_rule((('agproduct',), (('usedFor', 'in', (('animal',), ())),)), ghost=True)
        model.add_rule((('bird',), (('typeOf', 'in', (('animal',), ())),)), ghost=True)
        model.add_rule((('bird',), (('typeOf', 'out', (('mammal',), ())),)), ghost=True)

        rule1 = Rule(('agproduct',), (('madeFrom', 'out', (('animal',), ())),))
        rule2 = Rule(('agproduct',), (('usedFor', 'in', (('animal',), ())),))
        rule3 = Rule(('bird',), (('typeOf', 'in', (('animal',), ())),))
        rule4 = Rule(('bird',), (('typeOf', 'out', (('mammal',), ())),))

        model.build_rule_graph()
        assert(model.rule_graph.number_of_nodes() == 0)

        merged = Model(graph)
        rule1.merge(rule2)
        rule3.merge(rule4)
        merged.add_rule(rule1, ghost=True)
        merged.add_rule(rule3, ghost=True)
        merged.build_rule_graph()

        assert(merged.rule_graph.number_of_nodes() == 0)

    def test_plant_forest_1(self):
        '''
        Tests that a planting a forest leads to the correct number of correct_assertions
        and these cover the right number of edges.
        '''
        graph = Graph('test', idify=False, verbose=False)
        searcher = Searcher(graph)
        model = searcher.build_model(verbosity=0)
        rule = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        model.plant_forest(rule)
        assert(len(rule.correct_assertions) == 1)
        assert(len(rule.get_edges_covered()) == 6)
        assert(rule.get_labels_covered() == {('7241965', '36240'), ('7241965', '6555563'), ('7241965', '2415820'), ('7241965', '879961'), ('7241965', '6341376'), ('7241965', '6175574')})

    def test_plant_forest_2(self):
        '''
        Tests that a planting a forest leads to the correct number of correct_assertions
        and these cover the right number of edges.
        '''
        graph = Graph('test', idify=False, verbose=False)
        model = Model(graph)
        rule = Rule(('7241965',), (('6293378', 'in', (('1927286',), ())),))
        model.plant_forest(rule)
        assert(len(rule.correct_assertions) == 6)
        assert(len(rule.get_edges_covered()) == 6)
        assert(rule.get_labels_covered() == {('1927286', '7499850')})

    def test_plant_forest_3(self):
        '''
        Tests that a planting a forest leads to the correct number of correct_assertions
        and these cover the right number of labels.
        '''
        graph = Graph('test', idify=False, verbose=False)
        model = Model(graph)
        rule = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model.plant_forest(rule)
        assert(len(rule.correct_assertions) == 2)
        assert(rule.get_labels_covered() == {('5794125', '308389')})

    def test_rule_1(self):
        rule = (('1927286',), (('6293378', 'out', (('7241965',), ())),))
        parent, children = rule
        rule = Rule(parent, children)
        assert(rule.root == ('1927286',))
        assert(len(rule.children) == 1)
        assert(rule.children[0][2].root == ('7241965',))
        assert(rule.tuplify() == (('1927286',), (('6293378', 'out', (('7241965',), ())),)))

    def test_rule_coverage_1(self):
        graph = Graph('test', idify=False, verbose=False)
        model = Model(graph)
        rule = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        model.plant_forest(rule)
        assert(len(rule.correct_assertions) == 1)
        assert(len(rule.children) == 1)

        assert(len(rule.get_edges_covered()) == 6)
        assert(len(rule.get_labels_covered()) == 6)

    def test_level_1_model_1(self):
        graph = Graph('test', idify=False, verbose=False)
        model = Model(graph)
        assert(model.nest_rules(verbosity=0) == model)

    def test_level_1_model_2(self):
        graph = Graph('test', idify=False, verbose=False)
        searcher = Searcher(graph)
        model = searcher.build_model(verbosity=0)
        assert(model.nest_rules(verbosity=0) == model)

    def test_level_1_model_3(self):
        graph = Graph('test', idify=False, verbose=False)
        searcher = Searcher(graph)
        model = searcher.build_model_top_k_freq(k=10)
        assert(searcher.evaluator.evaluate(model.nest_rules(verbosity=0)) <= searcher.evaluator.evaluate(model))

    def test_merge_rules_1(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('1927286',), (('412681', 'out', (('7490702',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('1927286',), (('412681', 'out', (('7490702',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        merged = model.merge_rules(verbosity=0)
        rules = merged.rules
        assert(len(rules) == 1)
        for rule in rules:
            assert(len(rule.children) == 2)
            assert(set(child[0] for child in rule.children) == {'412681', '6293378'})
            assert(set(child[2].root for child in rule.children) == {('7241965',), ('7490702',)})
            assert(len(rule.correct_assertions) == 1)
            assert(set(rule.correct_assertions[0].nodes.keys()) == {('7499850'), ('36240'), ('6175574'), ('2415820'), ('879961'), ('6555563'), ('6341376'), ('8220493')})
            assert(len(rule.correct_assertions[0].nodes['7499850'].neighbors_of_type) == 2)

        evaluator = Evaluator(graph)
        assert(evaluator.evaluate(merged) < evaluator.evaluate(model))

    def test_merge_rules_2(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('1927286',), (('412681', 'out', (('7490702',), ())),)))
        model.add_rule((('1927286',), (('3320538', 'out', (('8226812',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('1927286',), (('412681', 'out', (('7490702',), ())),))
        rule3 = Rule(('1927286',), (('3320538', 'out', (('8226812',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        model.plant_forest(rule3)
        assert(rule1.merge(rule2))
        assert(rule1.merge(rule3))
        merged = model.merge_rules(verbosity=0)
        rules = merged.rules
        assert(len(model.rules) == 3)
        assert(len(rules) == 1)
        # should be full clique
        assert(model.shared_root_rule_dependency_graph.number_of_nodes() == 3)
        assert(model.shared_root_rule_dependency_graph.number_of_edges() == 3)
        assert(len(rules) == 1)
        for rule in rules:
            assert(len(rule.children) == 3)
            assert(set(child[0] for child in rule.children) == {'412681', '6293378', '3320538'})
            assert(set(child[2].root for child in rule.children) == {('7241965',), ('7490702',), ('8226812',)})
            assert(len(rule.correct_assertions) == 1)
            assert(set(rule.correct_assertions[0].nodes.keys()) == {('7499850'), ('36240'), ('6175574'), ('2415820'), ('879961'), ('6555563'), ('6341376'), ('8220493'), ('9054900')})
            assert(len(rule.correct_assertions[0].nodes['7499850'].neighbors_of_type) == 3)

        evaluator = Evaluator(graph)
        assert(evaluator.evaluate(merged) < evaluator.evaluate(model))

    def test_merge_rules_3(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('1927286',), (('412681', 'out', (('7490702',), ())),)))
        model.add_rule((('1927286',), (('3320538', 'out', (('8226812',), ())),)))
        model.add_rule((('1927286',), (('6291253', 'out', (('8226812',), ())),)))
        merged = model.merge_rules(verbosity=0)
        rules = merged.rules
        assert(len(model.rules) == 4)
        assert(len(rules) == 1)
        # should be full clique
        assert(model.shared_root_rule_dependency_graph.number_of_nodes() == 4)
        assert(model.shared_root_rule_dependency_graph.number_of_edges() == 6)
        for rule in rules:
            assert(len(rule.children) == 4)
            assert(set(child[0] for child in rule.children) == {'412681', '6293378', '3320538', '6291253'})
            assert(set(child[2].root for child in rule.children) == {('7241965',), ('7490702',), ('8226812',)})
            assert(len(rule.correct_assertions) == 1)
            assert(set(rule.correct_assertions[0].nodes.keys()) == {('7499850'), ('36240'), ('6175574'), ('2415820'), ('879961'), ('6555563'), ('6341376'), ('8220493'), ('9054900'), ('7992351')})
            assert(len(rule.correct_assertions[0].nodes['7499850'].neighbors_of_type) == 4)

        evaluator = Evaluator(graph)
        assert(evaluator.evaluate(merged) < evaluator.evaluate(model))

        gt_edges = set()
        for rule in model.rules:
            for edge in graph.candidates[rule]['edges']:
                gt_edges.add(edge)
        edges = set()
        for rule in rules:
            edges.update(set(rule.get_edges_covered()))
        assert(gt_edges == edges)

    def test_merge_rules_4(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('1927286',), (('412681', 'out', (('7490702',), ())),)))
        model.add_rule((('1927286',), (('3320538', 'out', (('8226812',), ())),)))
        model.add_rule((('1927286',), (('6291253', 'out', (('8226812',), ())),)))

    def test_merging_graph_1(self):
        graph = Graph('test', idify=False, verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        model.build_rule_graph()
        rules = sorted(model.rule_graph.nodes(), key=lambda it: it.root)
        r1, r2 = rules

        _graph = nx.DiGraph(model.rule_graph.edges())
        assert(_graph.number_of_nodes() == 2)
        assert(_graph.number_of_edges() == 1)

        candidate = deepcopy(r1)
        assert(candidate.pin_to_leaf(r2))

        _graph = nx.algorithms.minors.contracted_nodes(_graph, r1, r2, self_loops=False)

        assert(_graph.number_of_nodes() == 1)
        assert(_graph.number_of_edges() == 0)

        _graph = nx.relabel_nodes(_graph, {r1: candidate})

        assert(_graph.number_of_nodes() == 1)
        assert(_graph.number_of_edges() == 0)
        assert(candidate in _graph.nodes())

    def test_merging_graph_2(self):
        graph = Graph('test', idify=False, verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('3320538', 'out', (('8226812',), ())),)))
        model.add_rule((('7241965',), (('412681', 'out', (('7490702',), ())),)))
        model.build_rule_graph()
        for rule in model.rule_graph.nodes():
            if rule.children[0][0] == '6293378':
                r1 = rule
            elif rule.children[0][0] == '3320538':
                r2 = rule
            else:
                r3 = rule

        _graph = nx.DiGraph(model.rule_graph.edges())
        assert(_graph.number_of_nodes() == 3)
        assert(_graph.number_of_edges() == 2)

        candidate1 = deepcopy(r1)
        assert(candidate1.pin_to_leaf(r2))

        _graph = nx.algorithms.minors.contracted_nodes(_graph, r1, r2, self_loops=False)

        assert(_graph.number_of_nodes() == 2)
        assert(_graph.number_of_edges() == 1)

        _graph = nx.relabel_nodes(_graph, {r1: candidate1})

        assert(_graph.number_of_nodes() == 2)
        assert(_graph.number_of_edges() == 1)
        assert(candidate1 in _graph.nodes())

        candidate2 = deepcopy(candidate1)
        assert(candidate2.pin_to_leaf(r3))

        _graph = nx.algorithms.minors.contracted_nodes(_graph, candidate1, r3, self_loops=False)

        assert(_graph.number_of_nodes() == 1)
        assert(_graph.number_of_edges() == 0)

        _graph = nx.relabel_nodes(_graph, {candidate1: candidate2})

        assert(_graph.number_of_nodes() == 1)
        assert(_graph.number_of_edges() == 0)
        assert(candidate2 in _graph.nodes())

    def test_merging_graph_3(self):
        graph = Graph('test', idify=False, verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),))) # r1
        model.add_rule((('7241965',), (('3320538', 'out', (('8226812',), ())),))) # r2
        model.add_rule((('7241965',), (('412681', 'out', (('7490702',), ())),))) # r4
        model.add_rule((('3029686',), (('7051738', 'in', (('7241965',), ())),))) # r3
        model.build_rule_graph()
        for rule in model.rule_graph.nodes():
            if rule.children[0][0] == '6293378':
                r1 = rule
            elif rule.children[0][0] == '3320538':
                r2 = rule
            elif rule.children[0][0] == '7051738':
                r3 = rule
            else:
                r4 = rule

        _graph = nx.DiGraph(model.rule_graph.edges())
        assert(_graph.number_of_nodes() == 4)
        assert(_graph.number_of_edges() == 4)

        candidate1 = deepcopy(r1)
        assert(candidate1.pin_to_leaf(r2))
        _graph = nx.algorithms.minors.contracted_nodes(_graph, r1, r2, self_loops=False)
        _graph = nx.relabel_nodes(_graph, {r1: candidate1})
        to_remove = list()
        # not all are legal
        for x, y in _graph.out_edges(candidate1):
            if y.root not in x.get_inner_nodes():
                to_remove.append((x, y))
        _graph.remove_edges_from(to_remove)
        assert(_graph.number_of_nodes() == 3)
        assert(_graph.number_of_edges() == 3)

        candidate2 = deepcopy(r3)
        assert(candidate2.pin_to_leaf(r4))
        _graph = nx.algorithms.minors.contracted_nodes(_graph, r3, r4, self_loops=False)
        _graph = nx.relabel_nodes(_graph, {r3: candidate2})
        to_remove = list()
        assert(_graph.number_of_nodes() == 2)
        assert(_graph.number_of_edges() == 2)
        to_remove = list()
        # not all are legal
        for x, y in _graph.out_edges(candidate2):
            if y.root not in x.get_inner_nodes():
                to_remove.append((x, y))
        _graph.remove_edges_from(to_remove)
        assert(_graph.number_of_nodes() == 2)
        assert(_graph.number_of_edges() == 1)

    def test_label_qualify_1(self):
        '''
        Test that merge works when it should.
        '''
        graph = Graph('label_qualifier_1', verbose=False)
        searcher = Searcher(graph)
        assert((('green',), (('e1', 'out', (('blue',), ())),)) in graph.candidates)
        searcher.label_qualify(verbosity=0)
        assert((('green',), (('e1', 'out', (('blue',), ())),)) not in graph.candidates)
        assert((('green', 'red'), (('e1', 'out', (('blue',), ())),)) in graph.candidates)

    def test_label_qualify_2(self):
        '''
        Test that merge does not work when it doesn't save cost.
        '''
        graph = Graph('test', verbose=False)
        searcher = Searcher(graph)
        searcher.label_qualify(verbosity=0)
        assert((('7241965',), (('6293378', 'in', (('1927286',), ())),)) in graph.candidates)

if __name__ == "__main__":
    unittest.main()
