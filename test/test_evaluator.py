import unittest
import sys
sys.path.append('../src/')
from graph import Graph
from evaluator import Evaluator
from searcher import Searcher
from model import Model
from rule import Rule
from math import log2 as log
from scipy.special import comb
import numpy as np
from collections import defaultdict

class TestEvaluator(unittest.TestCase):

    def test_length_natural_number_0(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        assert(evaluator.length_natural_number(0) == 0)

    def test_length_natural_number_1(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        c = log(2.865064)
        assert(evaluator.length_natural_number(1) == c)

    def test_length_natural_number_2(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        c = log(2.865064)
        assert(evaluator.length_natural_number(1) == c)
        # make sure cache works
        evaluator.length_natural_number(2)
        assert(evaluator.length_natural_number_map[1] == c)
        assert(evaluator.length_natural_number(1) == c)
        evaluator.length_natural_number(3)
        assert(evaluator.length_natural_number(1) == c)
        assert(evaluator.length_natural_number_map[1] == c)
        evaluator.length_natural_number(4)
        assert(evaluator.length_natural_number(1) == c)
        assert(evaluator.length_natural_number_map[1] == c)

    def test_length_model_null(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        model = Model(graph)
        assert(evaluator.length_model(model) == log(2 * graph.num_node_labels * graph.num_edge_labels * graph.num_node_labels + 1))

    def test_length_rule_1(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule = (('1927286',), (('6293378', 'out', (('7241965',), ())),))
        model = Model(graph)
        model.add_rule(rule)
        # num labels
        length = log(graph.num_node_labels)
        # length labels
        length += -log(1 / graph.n)
        # num branches
        length += evaluator.length_natural_number(1 + 1)
        # edge dir
        length += 1
        # length edge label
        length += -log(7 / graph.m)
        # child
        length += log(graph.num_node_labels)
        length += -log(6 / graph.n)
        length += evaluator.length_natural_number(0 + 1)
        assert(np.abs(evaluator.length_rule(rule) - length) < 0.00001)

    def test_length_rule_2(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule = (('1927286',), (('6293378', 'out', (('7241965', '6843923'), ())),))
        # since we haven't implemented multiple label rules, add it manually
        graph.candidates[rule] = {'label_coverage': set(),
                                  'edges': set(),
                                  'ca_to_size': defaultdict(int)}

        for node in ['36240', '6175574', '6341376', '2415820', '6555563', '879961']:
            graph.candidates[rule]['label_coverage'].add((('7241965', '6843923'), node))
            graph.candidates[rule]['edges'].add(('7499850', node))
        graph.candidates[rule]['label_coverage'].add(('1927286', '7499850'))
        graph.candidates[rule]['ca_to_size']['7499850'] = 6

        model = Model(graph)
        model.add_rule(rule)
        # num labels
        length = log(graph.num_node_labels)
        # length labels
        length += -log(1 / graph.n)
        # num branches
        length += evaluator.length_natural_number(1 + 1)
        # edge dir
        length += 1
        # length edge label
        length += -log(7 / graph.m)
        # child
        length += log(graph.num_node_labels)
        length += -log(6 / graph.n)
        length += -log(7 / graph.n)
        length += evaluator.length_natural_number(0 + 1)
        assert(np.abs(evaluator.length_rule(rule) - length) < 0.00001)

    def test_length_rule_matches_assertions_1(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule = (('1927286',), (('6293378', 'out', (('7241965',), ())),))
        model = Model(graph)
        model.add_rule(rule)
        # cas (one)
        length = log(graph.n) + evaluator.length_binomial(graph.n - 1, 6)
        assert(evaluator.length_rule_assertions(rule, model) == length)

    def test_length_rule_matches_assertions_2(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule = (('1927286',), (('6293378', 'out', (('7241965', '6843923'), ())),))

        # since we haven't implemented multiple label rules, add it manually
        graph.candidates[rule] = {'label_coverage': set(),
                                  'edges': set(),
                                  'ca_to_size': defaultdict(int)}

        for node in ['36240', '6175574', '6341376', '2415820', '6555563', '879961']:
            graph.candidates[rule]['label_coverage'].add((('7241965', '6843923',), node))
            graph.candidates[rule]['edges'].add(('7499850', node))
        graph.candidates[rule]['label_coverage'].add(('1927286', '7499850'))
        graph.candidates[rule]['ca_to_size']['7499850'] = 6

        model = Model(graph)
        model.add_rule(rule)

        # TODO: won't work until we implement multiple labels
        # assert(evaluator.length_rule_assertions(rule, model) == length)

    def test_length_rule_matches_assertions_3(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule = (('1927286',), (('6293378', 'out', (('6843923',), ())),))
        model = Model(graph)
        model.add_rule(rule)
        # cas (one)
        length = log(graph.n) + evaluator.length_binomial(graph.n - 1, 7)
        assert(np.abs(evaluator.length_rule_assertions(rule, model) - length) < 0.00001)

    def test_length_model_1(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule = (('1927286',), (('6293378', 'out', (('7241965',), ())),))
        model = Model(graph)
        model.add_rule(rule)

        # num rules
        length = log(2 * graph.num_node_labels * graph.num_edge_labels * graph.num_node_labels + 1)

        # length of rule

        # num labels
        length += log(graph.num_node_labels)
        # length labels
        length += -log(1 / graph.n)
        # num branches
        length += evaluator.length_natural_number(1 + 1)
        # edge dir
        length += 1
        # length edge label
        length += -log(7 / graph.m)
        # child
        length += log(graph.num_node_labels)
        length += -log(6 / graph.n)
        length += evaluator.length_natural_number(0 + 1)

        # length of assertions

        # cas (one)
        length += log(graph.n) + evaluator.length_binomial(graph.n - 1, 6)

        assert(np.abs(evaluator.length_model(model) - length) < 0.00001)

    def test_length_model_2(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule = (('7241965',), (('7051738', 'out', (('3029686',), ())),))
        model = Model(graph)
        model.add_rule(rule)

        # num rules
        length = log(2 * graph.num_node_labels * graph.num_edge_labels * graph.num_node_labels + 1)

        # length of rule

        # num labels
        length += log(graph.num_node_labels)
        # length labels
        length += -log(6 / graph.n)
        # num branches
        length += evaluator.length_natural_number(1 + 1)
        # edge dir
        length += 1
        # length edge label
        length += -log(2 / graph.m)
        # child
        length += log(graph.num_node_labels)
        length += -log(1 / graph.n)
        length += evaluator.length_natural_number(0 + 1)

        # length of matches

        # num exceptions
        length += log(6)
        # exception ids
        length += evaluator.length_binomial(6, 4)
        # correct assertions
        length += log(graph.n) + evaluator.length_binomial(graph.n - 1, 1)
        length += log(graph.n) + evaluator.length_binomial(graph.n - 1, 1)

        assert(np.abs(evaluator.length_model(model) - length) <= 0.000001)

    def test_negative_edge_error_1(self):
        '''
        null model
        '''
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        model = Model(graph)

        num_each_edge = [7, 2, 3, 3, 4, 2, 2]
        length = evaluator.length_binomial(graph.n ** 2 * graph.num_edge_labels, sum(num_each_edge))

        assert(evaluator.length_negative_edge_error(model) == length)

    def test_negative_edge_error_3(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule1 = (('1927286',), (('6293378', 'out', (('6843923',), ())),))
        rule2 = (('7241965',), (('7946920', 'out', (('8359357',), ())),))
        model = Model(graph)
        model.add_rule(rule1)
        model.add_rule(rule2)

        num_each_edge = [3, 3, 4, 2, 2]
        length = evaluator.length_binomial(graph.n ** 2 * graph.num_edge_labels - 9, sum(num_each_edge))

        assert(evaluator.length_negative_edge_error(model) == length)

    def test_negative_node_error_1(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule = (('1927286',), (('6293378', 'out', (('6843923',), ())),))
        model = Model(graph)
        model.add_rule(rule)

        length = evaluator.length_binomial(graph.num_node_labels * graph.n - 7, graph.total_num_labels - 7)

        assert(np.abs(evaluator.length_negative_label_error(model) - length) < 0.000001)

    def test_negative_node_error_2(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule1 = (('1927286',), (('6293378', 'out', (('6843923',), ())),))
        rule2 = (('3029686',), (('7051738', 'in', (('7241965',), ())),))
        model = Model(graph)
        model.add_rule(rule1)
        model.add_rule(rule2)

        length = evaluator.length_binomial(graph.num_node_labels * graph.n - 9, graph.total_num_labels - 9)

        assert(np.abs(evaluator.length_negative_label_error(model) - length) < 0.000001)

    def test_negative_node_error_3(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule1 = (('1927286',), (('6293378', 'out', (('6843923',), ())),))
        rule2 = (('8359357',), (('7946920', 'in', (('7241965',), ())),))
        rule3 = (('1927286',), (('3320538', 'out', (('5266930',), ())),))
        rule4 = (('3029686',), (('7051738', 'in', (('7241965',), ())),))
        rule5 = (('6057655',), (('7051738', 'in', (('7241965',), ())),))
        rule6 = (('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model = Model(graph)
        model.add_rule(rule1)
        model.add_rule(rule2)
        model.add_rule(rule3)
        model.add_rule(rule4)
        model.add_rule(rule5)
        model.add_rule(rule6)

        assert(model.label_matrix == ({('6843923', '36240'),
                                       ('6843923', '6341376'),
                                       ('6843923', '919756'),
                                       ('6843923', '879961'),
                                       ('6843923', '2415820'),
                                       ('6843923', '6175574'),
                                       ('6843923', '6555563'),
                                       ('7241965', '6175574'),
                                       ('5266930', '9054900'),
                                       ('7241965', '6341376'),
                                       ('7241965', '879961'),
                                       ('5794125', '308389')}))

        length = evaluator.length_binomial(graph.num_node_labels * graph.n - 12, graph.total_num_labels - 12)

        assert(np.abs(evaluator.length_negative_label_error(model) - length) < 0.000001)

    def test_length_all_1(self):
        '''
        '''
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule = (('1927286',), (('6293378', 'out', (('7241965',), ())),))
        null_model = Model(graph)
        model = Model(graph)
        model.add_rule(rule)

        assert(evaluator.evaluate(model) < evaluator.evaluate(null_model))

    def test_length_all_3(self):
        '''
        Adding a rule that doesn't make error shouldn't cause the error to go up.
        '''
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        null_model = Model(graph)
        searcher = Searcher(graph)
        for rule in searcher.candidates:
            model = Model(graph)
            model.add_rule(rule)
            assert(evaluator.length_graph_with_model(model) <= evaluator.length_graph_with_model(null_model) or graph.nodes_with_type(rule[0]) - len(graph.candidates[rule]['ca_to_size']) > 0)

    def test_evaluate_1(self):
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        null_model = Model(graph)
        model = Model(graph)
        rule = (('1927286',), (('6293378', 'out', (('6843923',), ())),))
        before_rule = evaluator.length_negative_edge_error(null_model)
        model.add_rule(rule)
        as_error = before_rule - evaluator.length_negative_edge_error(model)
        as_star = evaluator.length_rule_assertions(rule, model)
        assert(as_star < as_error)

    def test_evaluate_change_1(self):
        '''
        '''
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule = (('1927286',), (('6293378', 'out', (('7241965',), ())),))
        model = Model(graph)
        # before rule
        _, length_model, neg_edge, neg_node = evaluator.evaluate(model, with_lengths=True)
        assert(length_model >= 0)
        assert(neg_edge >= 0)
        assert(neg_node >= 0)
        model.add_rule(rule)
        # after rule
        gt_val = evaluator.evaluate(model)
        new_val, _, new_neg_edge, new_neg_node = evaluator.evaluate_change(model, rule, length_model)
        assert(new_neg_edge < neg_edge)
        assert(new_neg_node < neg_node)
        assert(gt_val == new_val)

    def test_evaluate_change_2(self):
        '''
        '''
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        rule1 = (('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = (('3029686',), (('7051738', 'in', (('7241965',), ())),))
        model = Model(graph)
        gt = Model(graph)
        gt.add_rule(rule1)

        _, length_model, neg_edge, neg_node = evaluator.evaluate(model, with_lengths=True)
        val = evaluator.evaluate(gt)
        model.add_rule(rule1)
        new_val, length_model, neg_edge, neg_node = evaluator.evaluate_change(model, rule1, length_model)
        assert(val == new_val)

        gt.add_rule(rule2)
        val = evaluator.evaluate(gt)
        model.add_rule(rule2)
        new_val = evaluator.evaluate_change(model, rule2, length_model)[0]
        assert(val == new_val)

    def test_length_null_1(self):
        '''
        test the length of a null model
        '''
        graph = Graph('test', verbose=False)
        evaluator = Evaluator(graph)
        model = Model(graph)
        val = evaluator.evaluate(model)
        err = evaluator.length_graph_with_model(model)[0]
        err_b = evaluator.length_negative_edge_error(model) + evaluator.length_negative_label_error(model)
        assert(evaluator.length_model(model) == log(2 * graph.num_node_labels * graph.num_edge_labels * graph.num_node_labels + 1))
        assert(val - log(2 * graph.num_node_labels * graph.num_edge_labels * graph.num_node_labels + 1) == err == err_b)

    def test_idified(self):
        graph = Graph('test', idify=True, verbose=False)
        evaluator = Evaluator(graph)
        rule = ((graph.label_to_id['1927286'],), ((graph.pred_to_id['6293378'], 'out', ((graph.label_to_id['7241965'],), ())),))
        model = Model(graph)
        model.add_rule(rule)
        evaluator.evaluate(model)

    def test_length_binomial_1(self):
        graph = Graph('test', idify=True, verbose=False)
        evaluator = Evaluator(graph)

        assert(log(comb(5, 2)) == evaluator.length_binomial(5, 2))
        assert(np.abs(log(comb(15, 4)) - evaluator.length_binomial(15, 4)) <= 0.00001)
        assert(np.abs(log(comb(13, 13)) - evaluator.length_binomial(13, 13)) <= 0.00001)
        assert(np.abs(log(comb(646, 1)) - evaluator.length_binomial(646, 1)) <= 0.00001)
        assert(np.abs(log(comb(463, 35)) == evaluator.length_binomial(463, 35)) <= 0.00001)

    def test_level_1_rules_1(self):
        graph = Graph('test', idify=False, verbose=False)
        evaluator = Evaluator(graph)
        level_0_model = Model(graph)
        rule1 = (('1927286',), (('6293378', 'out', (('7241965',), ())),))
        level_0_model.add_rule(rule1)
        level_0_score = evaluator.evaluate(level_0_model)
        level_1_model = Model(graph)
        rule2 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        level_1_model.plant_forest(rule2)
        level_1_model.add_rule(rule2)
        level_1_score = evaluator.evaluate(level_1_model)
        assert(evaluator.length_rule(rule1) == evaluator.length_rule(rule2))
        assert(evaluator.length_rule_assertions(rule1, level_0_model) == evaluator.length_rule_assertions(rule2, level_1_model))
        assert(evaluator.length_model(level_0_model) == evaluator.length_model(level_1_model))
        assert(level_0_score == level_1_score)

    def test_rule_to_length_1(self):
        graph = Graph('test', verbose=False)
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model = Model(graph)
        model.add_rule(rule1)
        model.add_rule(rule2)
        model.plant_forest(rule1)
        model.plant_forest(rule2)

if __name__ == "__main__":
    unittest.main()
