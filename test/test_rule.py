import unittest
import sys
sys.path.append('../src/')
from graph import Graph
from model import Model
from searcher import Searcher
from evaluator import Evaluator
from scipy.sparse import csr_matrix
import numpy as np
from rule import Rule
from correct_assertion import CorrectAssertion
from copy import deepcopy

class TestRule(unittest.TestCase):
    def test_rule_init(self):
        assert(Rule('root'))

    def test_pin_to_leaf_1(self):
        rule = Rule('root', children=[('pred', 'out', ('leaf', ()))])
        leaf = Rule('leaf', children=[('other_pred', 'out', ('new_leaf', ()))])
        assert(rule.get_leaves() == ['leaf'])
        assert(rule.pin_to_leaf(leaf))
        assert(rule.get_leaves() == ['new_leaf'])

    def test_pin_to_leaf_2(self):
        '''
        Test that pinning to a leaf works when there is a match.
        '''
        rule = Rule(('root',), children=[('pred', 'out', (('leaf',), ()))])
        leaf = Rule(('leaf',), children=[('other_pred', 'out', (('new_leaf',), ()))])
        ca1 = CorrectAssertion('root1', label=('root',))
        # (u, u_typ, pred, dir, v, v_typ)
        ca1.add_edge(('root1', ('root',), 'pred', 'out', 'leaf1', ('leaf',)), labels=True)
        rule.insert_correct_assertion(ca1)
        assert(rule.instantiated())
        ca2 = CorrectAssertion(('leaf1',), label=('leaf',))
        ca2.add_edge(('new_leaf1', ('leaf',), 'other_pred', 'out', 'leaf1', ('new_leaf',)), labels=True)
        leaf.insert_correct_assertion(ca2)
        assert(leaf.instantiated())
        assert(rule.get_leaves() == [('leaf',)])
        assert(rule.pin_to_leaf(leaf))
        assert(rule.get_leaves() == [('new_leaf',)])
        assert(len(rule.correct_assertions) == 1)
        for ca in rule.correct_assertions:
            assert(len(ca.nodes[ca.root].neighbors) == 1)
            assert(len(ca.nodes[ca.root].neighbors_of_type) == 1)

    def test_pin_to_leaf_3(self):
        '''
        Test that pinning to a leaf does not work when there is no match.
        '''
        rule = Rule('root', children=[('pred', 'out', ('leaf', ()))])
        leaf = Rule('some6057655_other_than_leaf', children=[('other_pred', 'out', ('new_leaf', ()))])
        assert(rule.get_leaves() == ['leaf'])
        assert(not rule.pin_to_leaf(leaf))
        assert(rule.get_leaves() == ['leaf'])

    def test_pin_to_leaf_4(self):
        '''
        Test that pinning to a leaf works when there is a match, but there are multiple options to search over.
        '''
        rule = Rule('root', children=[('pred', 'out', ('leaf', ()))])
        leaf = Rule('leaf', children=[('other_pred', 'out', ('new_leaf', ()))])
        ca1 = CorrectAssertion('root1', label='root')
        ca2 = CorrectAssertion('root2', label='root')
        ca1.add_edge(('root1', ('root',), 'pred', 'out', 'leaf1', ('leaf',)), labels=True)
        ca2.add_edge(('root2', ('root',), 'other_pred', 'out', 'leaf2', ('leaf',)), labels=True)
        rule.insert_correct_assertion(ca1)
        rule.insert_correct_assertion(ca2)
        ca3 = CorrectAssertion('leaf2', label='leaf')
        ca3.add_edge(('leaf2', ('root',), 'pred', 'out', 'new_leaf1', ('new_leaf',)), labels=True)
        leaf.insert_correct_assertion(ca3)
        assert(rule.get_leaves() == ['leaf'])
        assert(rule.pin_to_leaf(leaf))
        assert(rule.get_leaves() == ['new_leaf'])

    def test_pin_to_leaf_5(self):
        '''
        Test that pinning to a leaf correct assertions works when there is a match.
        '''
        rule = Rule('root', children=[('pred', 'out', ('leaf', ()))])
        leaf = Rule('leaf', children=[('other_pred', 'out', ('new_leaf', ()))])
        ca1 = CorrectAssertion('root1', label='root')
        ca1.add_edge(('root1', ('root',), 'pred', 'out', 'leaf1', ('leaf',)), labels=True)
        rule.insert_correct_assertion(ca1)
        assert(rule.instantiated())
        ca2 = CorrectAssertion('leaf1', label='leaf')
        ca2.add_edge(('leaf1', ('root',), 'pred', 'out', 'new_leaf1', ('new_leaf',)), labels=True)
        leaf.insert_correct_assertion(ca2)
        assert(leaf.instantiated())
        assert(rule.get_leaves() == ['leaf'])
        assert(rule.pin_to_leaf(leaf))

    def test_pin_to_leaf_6(self):
        '''
        Test that pinning to a leaf correct_assertion works on the ca graph.
        '''
        model = Model(Graph('test', idify=False, verbose=False))
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        assert(rule1.pin_to_leaf(rule2))

    def test_pin_to_leaf_7(self):
        '''
        Test that pinning to a leaf correct_assertion works on the ca graph.
        '''
        model = Model(Graph('test', idify=False, verbose=False))
        model.add_rule((('7490702',), (('412681', 'in', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('6293378', 'in', (('1927286',), ())),)))
        rule1 = Rule(('7490702',), (('412681', 'in', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('6293378', 'in', (('1927286',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        for ca in rule2.correct_assertions:
            res = None
            for r in rule1.correct_assertions:
                res = ca.root in r.nodes or res
            if ca.root in {'36240', '6555563', '6175574'}:
                assert(res)
            else:
                assert(not res)
        assert(rule1.pin_to_leaf(rule2))


    def test_pin_to_leaf_8(self):
        '''
        Test that pinning multiple times to a leaf doesn't overwrite the leaf.
        '''
        model = Model(Graph('test', idify=False, verbose=False))
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('412681', 'out', (('7490702',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('412681', 'out', (('7490702',), ())),))
        rule3 = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        model.plant_forest(rule3)
        assert(rule1.pin_to_leaf(rule2))
        assert(rule1.pin_to_leaf(rule3))
        assert(len(rule1.children) == 1)
        assert(set(child[2].root for child in rule1.children[0][2].children) == {('5794125',), ('7490702',)})

    def test_pin_to_leaf_9(self):
        '''
        Test that pinning works in tiny toy graph.
        '''
        model = Model(Graph('tiny', verbose=False))
        model.add_rule((('green',), (('black', 'out', (('blue',), ())),)))
        model.add_rule((('blue',), (('other_black', 'out', (('red',), ())),)))
        rule1 = Rule(('green',), (('black', 'out', (('blue',), ())),))
        rule2 = Rule(('blue',), (('other_black', 'out', (('red',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        assert(len(rule1.correct_assertions) == 2)
        assert(len(rule2.correct_assertions) == 4)

        assert(rule1.pin_to_leaf(rule2))
        assert(len(rule1.correct_assertions) == 2)
        for ca, root in zip(rule1.correct_assertions, ['1', '2']):
            assert(ca.root == root)
            assert(len(ca.nodes[ca.root].neighbors) == 2)
            assert(set(ca.nodes[ca.root].neighbors_of_type.keys()) == {('black', 'out', ('blue',))})
            if ca.root == '1':
                assert(set(ca.nodes[ca.root].neighbors_of_type[('black', 'out', ('blue',))]) == {'3', '4'})
            else:
                assert(set(ca.nodes[ca.root].neighbors_of_type[('black', 'out', ('blue',))]) == {'5', '6'})
            for child in ca.nodes[ca.root].neighbors:
                child = ca.nodes[child]
                if child.name == '3':
                    assert(len(ca.nodes[child.name].neighbors) == 4)
                    assert(set(ca.nodes[child.name].neighbors_of_type.keys()) == {('black', 'in', ('green',)), ('other_black', 'out', ('red',))})
                    assert(set(ca.nodes[child.name].neighbors_of_type[('other_black', 'out', ('red',))]) == {'8', '9', '10'})
                elif child.name == '4':
                    assert(len(ca.nodes[child.name].neighbors) == 2)
                    assert(set(ca.nodes[child.name].neighbors_of_type.keys()) == {('black', 'in', ('green',)), ('other_black', 'out', ('red',))})
                    assert(set(ca.nodes[child.name].neighbors_of_type[('other_black', 'out', ('red',))]) == {'11'})
                elif child.name == '5':
                    assert(len(ca.nodes[child.name].neighbors) == 2)
                    assert(set(ca.nodes[child.name].neighbors_of_type.keys()) == {('black', 'in', ('green',)), ('other_black', 'out', ('red',))})
                    assert(set(ca.nodes[child.name].neighbors_of_type[('other_black', 'out', ('red',))]) == {'12'})
                elif child.name == '6':
                    assert(len(ca.nodes[child.name].neighbors) == 1)
                    assert(set(ca.nodes[child.name].neighbors_of_type.keys()) == {('black', 'in', ('green',)),})
                else:
                    assert(False)

    def test_pin_to_leaf_10(self):
        '''
        Test that pinning works in tiny1 toy graph.
        '''
        model = Model(Graph('tiny1', verbose=False))
        model.add_rule((('green',), (('black0', 'out', (('blue',), ())),)))
        model.add_rule((('blue',), (('black1', 'out', (('purple',), ())),)))
        model.add_rule((('blue',), (('black2', 'out', (('red',), ())),)))
        rule1 = Rule(('green',), (('black0', 'out', (('blue',), ())),))
        rule2 = Rule(('blue',), (('black1', 'out', (('purple',), ())),))
        rule3 = Rule(('blue',), (('black2', 'out', (('red',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        model.plant_forest(rule3)
        assert(rule1.pin_to_leaf(rule2))
        assert(rule1.pin_to_leaf(rule3))
        assert(set(ca.root for ca in rule1.correct_assertions) == {'1', '2'})
        for ca in rule1.correct_assertions:
            if ca.root == '1':
                assert(set(ca.nodes['1'].neighbors) == {'3', '4'})
                assert(set(ca.nodes['1'].neighbors_of_type.keys()) == {('black0', 'out', ('blue',))})
                for child in ca.nodes.keys():
                    if child == '3':
                        a = ca.nodes['3'].neighbors
                        b = set()
                        for cs in ca.nodes['3'].neighbors_of_type.values():
                            b.update(cs)
                        assert(len(a) == 3)
                        assert(len(b) == 3)
                        assert(a == b)
                        assert(a == {'1', '8', '9'})
                        assert(set(ca.nodes[child].neighbors_of_type.keys()) == {('black0', 'in', ('green',)), ('black1', 'out', ('purple',)), ('black2', 'out', ('red',))})
                    if child == '4':
                        assert(ca.nodes['4'].neighbors == {'1', '10', '11'})
                        assert(set(ca.nodes['4'].neighbors_of_type.keys()) == {('black0', 'in', ('green',)), ('black1', 'out', ('purple',)), ('black2', 'out', ('red',))})
                        a = set(ca.nodes['4'].neighbors)
                        b = set()
                        for cs in ca.nodes['4'].neighbors_of_type.values():
                            b.update(cs)
                        assert(len(a) == 3)
                        assert(len(b) == 3)
                        assert(a == b)
            elif ca.root == '2':
                assert(set(ca.nodes[ca.root].neighbors) == {'5', '6'})
                assert(set(ca.nodes[ca.root].neighbors_of_type.keys()) == {('black0', 'out', ('blue',))})
                for child in ca.nodes.keys():
                    if child == '5':
                        assert(ca.nodes['5'].neighbors == {'2', '12'})
                        assert(set(ca.nodes['5'].neighbors_of_type.keys()) == {('black0', 'in', ('green',)), ('black2', 'out', ('red',))})
                        a = ca.nodes['5'].neighbors
                        b = set()
                        for cs in ca.nodes['5'].neighbors_of_type.values():
                            b.update(cs)
                        assert(len(a) == 2)
                        assert(len(b) == 2)
                        assert(a == b)
                    if child == '6':
                        assert(ca.nodes['6'].neighbors == {'2', '13'})
                        assert(set(ca.nodes['6'].neighbors_of_type.keys()) == {('black0', 'in', ('green',)), ('black1', 'out', ('purple',))})
                        a = ca.nodes['6'].neighbors
                        b = set()
                        for cs in ca.nodes['6'].neighbors_of_type.values():
                            b.update(cs)
                        assert(len(a) == 2)
                        assert(len(b) == 2)
                        assert(a == b)

        assert(len(rule1.correct_assertions) == 2)
        rule1.filter_errant()
        assert(len(rule1.correct_assertions) == 1)
        assert(rule1.correct_assertions[0].root == '1')


    def test_insert_correct_assertion_1(self):
        '''
        Test that inserting a correct_assertion into a rule works.
        '''
        rule = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        ca = CorrectAssertion('7499850', label='1927286')
        ca.add_edge(('7499850', ('1927286',), '6293378', 'out', '36240', ('7241965',)))
        ca.add_edge(('7499850', ('1927286',), '6293378', 'out', '2415820', ('7241965',)))
        ca.add_edge(('7499850', ('1927286',), '6293378', 'out', '6175574', ('7241965',)))
        assert(len(ca.nodes['7499850'].neighbors) == 3)
        rule.insert_correct_assertion(ca)
        assert(rule.instantiated())
        assert(rule.correct_assertions[0] == ca)

    def test_correct_assertions_1(self):
        '''
        Checks that Rule objects have the same correct assertions as in the graph.
        '''
        graph = Graph('test', idify=False, verbose=False)
        model = Model(graph)
        for rule, cas in graph.candidates.items():
            model.add_rule(rule)
            tree = Rule(rule[0], rule[1])
            model.plant_forest(tree)
            assert(set(ca.root for ca in tree.correct_assertions) == set(cas['ca_to_size'].keys()))
            # if cas overlap, the correct assertion children will be fewer
            assert(len(tree.correct_assertions) <= sum(cas['ca_to_size'].values()))

    def test_filter_errant_1(self):
        model = Model(Graph('test', idify=False, verbose=False))
        model.add_rule((('8226812',), (('3320538', 'in', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('412681', 'out', (('7490702',), ())),)))
        rule1 = Rule(('8226812',), (('3320538', 'in', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('412681', 'out', (('7490702',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        assert(rule1.pin_to_leaf(rule2))
        assert(rule1.correct_assertions[0].root == '7992351')
        assert(rule1.correct_assertions[0].nodes['7992351'].neighbors == {'2415820'})
        assert(rule1.correct_assertions[0].nodes['7992351'].neighbors_of_type.keys() == {('3320538', 'in', ('7241965',))})
        assert(set(rule1.correct_assertions[0].nodes.keys()) == {'7992351', '2415820'})
        assert(rule1.correct_assertions[1].root == '2873925')
        assert(rule1.correct_assertions[1].nodes['2873925'].neighbors == {'36240'})
        assert(set(rule1.correct_assertions[1].nodes.keys()) == {'2873925', '36240', '3352101'})

        assert(len(rule1.correct_assertions) == 2)
        rule1.filter_errant()
        assert(len(rule1.correct_assertions) == 1)
        assert(rule1.correct_assertions[0].root == '2873925')

    def test_filter_errant_2(self):
        model = Model(Graph('test', idify=False, verbose=False))
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        assert(rule1.pin_to_leaf(rule2))
        assert(len(rule1.correct_assertions) == 1)
        assert(rule1.correct_assertions[0].root == '7499850')
        assert(set(rule1.correct_assertions[0].nodes.keys()) == {'7499850', '6175574', '36240', '6555563', '879961', '2415820', '6341376', '308389'})

        rule1.filter_errant()
        assert(len(rule1.correct_assertions) == 0)

    def test_filter_errant_3(self):
        rule1 = Rule(('green',), (('black', 'out', (('blue',), ())),))
        rule2 = Rule(('blue',), (('other_black', 'out', (('red',), ())),))

        ca1 = CorrectAssertion(1, label=('green',))
        # (u, u_typ, pred, dir, v, v_typ)
        ca1.add_edge((1, ('green',), 'black', 'out', 3, ('blue',)))
        ca1.add_edge((1, ('green',), 'black', 'out', 4, ('blue',)))

        ca2 = CorrectAssertion(2, label=('green',))
        ca2.add_edge((2, ('green',), 'black', 'out', 5, ('blue',)))
        ca2.add_edge((2, ('green',), 'black', 'out', 6, ('blue',)))

        ca3 = CorrectAssertion(3, label=('blue',))
        ca3.add_edge((3, ('blue',), 'other_black', 'out', 8, ('red',)))
        ca3.add_edge((3, ('blue',), 'other_black', 'out', 9, ('red',)))
        ca3.add_edge((3, ('blue',), 'other_black', 'out', 10, ('red',)))

        ca4 = CorrectAssertion(4, label=('blue',))
        ca4.add_edge((4, ('blue',), 'other_black', 'out', 11, ('red',)))

        ca5 = CorrectAssertion(5, label=('blue',))
        ca5.add_edge((5, ('blue',), 'other_black', 'out', 12, ('red',)))

        ca6 = CorrectAssertion(7, label=('blue',))
        ca6.add_edge((7, ('blue',), 'other_black', 'out', 13, ('red',)))

        rule1.insert_correct_assertion(ca1)
        rule1.insert_correct_assertion(ca2)

        rule2.insert_correct_assertion(ca3)
        rule2.insert_correct_assertion(ca4)
        rule2.insert_correct_assertion(ca5)
        rule2.insert_correct_assertion(ca6)

        assert(rule1.pin_to_leaf(rule2))

        assert(len(rule1.correct_assertions) == 2)
        rule1.filter_errant()
        assert(len(rule1.correct_assertions) == 1)

    def test_filter_errant_4(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        assert(rule1.pin_to_leaf(rule2))
        rule1.filter_errant()
        assert(len(rule1.correct_assertions) == 0)

    def test_filter_errant_5(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('3029686',), (('7051738', 'in', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('6293378', 'in', (('1927286',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        rule1 = Rule(('3029686',), (('7051738', 'in', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('6293378', 'in', (('1927286',), ())),))
        rule3 = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        model.plant_forest(rule3)
        assert(rule1.pin_to_leaf(rule2))
        assert(rule1.pin_to_leaf(rule3))
        rule1.filter_errant()
        assert(len(rule1.correct_assertions) == 0)

    def test_get_atoms_1(self):
        rule1 = Rule(('8226812',), (('3320538', 'in', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('412681', 'out', (('7490702',), ())),))
        tuple1 = rule1.tuplify()
        tuple2 = rule2.tuplify()
        assert(rule1.get_atoms()[0] == rule1.tuplify())
        assert(rule2.get_atoms()[0] == rule2.tuplify())
        assert(rule1.pin_to_leaf(rule2))
        assert(sorted(rule1.get_atoms()) == sorted((tuple1, tuple2)))

    def test_get_atoms_2(self):
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('1927286',), (('412681', 'out', (('7490702',), ())),))
        rule3 = Rule(('1927286',), (('3320538', 'out', (('8226812',), ())),))
        rule4 = Rule(('1927286',), (('6291253', 'out', (('8226812',), ())),))

        merged = deepcopy(rule1)
        merged.merge(rule2)
        merged.merge(rule3)
        merged.merge(rule4)
        assert(set(merged.get_atoms()) == set(rule.tuplify() for rule in [rule1, rule2, rule3, rule4]))

    def test_get_correct_at_depth_1(self):
        rule1 = Rule(('green',), (('black', 'out', (('blue',), ())),))
        rule2 = Rule(('blue',), (('other_black', 'out', (('red',), ())),))

    def test_get_correct_at_depth_2(self):
        graph = Graph('test', idify=False, verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)

    def test_get_assertions_at_depth_1(self):
        rule1 = Rule(('green',), (('black', 'out', (('blue',), ())),))
        rule2 = Rule(('blue',), (('other_black', 'out', (('red',), ())),))

    def test_get_assertions_at_depth_2(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)

    def test_get_assertions_at_depth_3(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('8226812',), (('3320538', 'in', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('412681', 'out', (('7490702',), ())),)))
        rule1 = Rule(('8226812',), (('3320538', 'in', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('412681', 'out', (('7490702',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)

    def test_rule_coverage_1(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        assert(len(rule1.get_edges_covered()) == 6)
        assert(len(rule1.get_labels_covered()) == 6)
        assert(len(rule2.get_edges_covered()) == 2)
        assert(len(rule2.get_labels_covered()) == 1)
        assert(rule1.pin_to_leaf(rule2))
        assert(len(rule1.get_edges_covered()) == 8)
        assert(len(rule1.get_labels_covered()) == 7)

    def test_rule_coverage_2(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('8226812',), (('3320538', 'in', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('412681', 'out', (('7490702',), ())),)))
        rule1 = Rule(('8226812',), (('3320538', 'in', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('412681', 'out', (('7490702',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        assert(len(rule1.get_edges_covered()) == 2)
        assert(len(rule1.get_labels_covered()) == 2)
        assert(len(rule2.get_edges_covered()) == 3)
        assert(len(rule2.get_labels_covered()) == 3)
        assert(rule1.pin_to_leaf(rule2))
        assert(len(rule1.get_edges_covered()) == 3)
        assert(len(rule1.get_labels_covered()) == 3)

        rule1.filter_errant()
        assert(len(rule1.get_edges_covered()) == 2)
        assert(len(rule1.get_labels_covered()) == 2)

    def test_rule_coverage_3(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        assert(len(rule1.get_edges_covered()) == 6)
        assert(len(rule1.get_labels_covered()) == 6)
        assert(len(rule2.get_edges_covered()) == 2)
        assert(len(rule2.get_labels_covered()) == 1)
        assert(rule1.pin_to_leaf(rule2))
        assert(len(rule1.get_edges_covered()) == 8)
        assert(len(rule1.get_labels_covered()) == 7)
        rule1.filter_errant()
        assert(len(rule1.correct_assertions) == 0)
        assert(len(rule1.get_edges_covered()) == 0)
        assert(len(rule1.get_labels_covered()) == 0)

    def test_rule_merge_1(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('1927286',), (('412681', 'out', (('7490702',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('1927286',), (('412681', 'out', (('7490702',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        original_covered_edges = set(rule1.get_edges_covered())
        original_covered_labels = set(rule1.get_labels_covered())
        assert(len(rule1.children) == 1)
        assert(len(rule1.correct_assertions[0].nodes['7499850'].neighbors_of_type) == 1)
        assert(rule1.merge(rule2))
        assert(len(rule1.children) == 2)
        assert(set(child[0] for child in rule1.children) == {'412681', '6293378'})
        assert(set(child[2].root for child in rule1.children) == {('7241965',), ('7490702',)})
        assert(len(rule1.correct_assertions) == 1)
        assert(rule1.get_edges_covered() == original_covered_edges.union(rule2.get_edges_covered()))
        assert(rule1.get_labels_covered() == original_covered_labels.union(rule2.get_labels_covered()))
        assert(len(rule1.correct_assertions[0].nodes['7499850'].neighbors_of_type) == 2)

    def test_rule_merge_2(self):
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
        original_covered_edges = set(rule1.get_edges_covered())
        original_covered_labels = set(rule1.get_labels_covered())
        assert(len(rule1.children) == 1)
        assert(len(rule1.correct_assertions[0].nodes['7499850'].neighbors_of_type) == 1)
        assert(rule1.merge(rule2))
        assert(len(rule1.children) == 2)
        assert(set(child[0] for child in rule1.children) == {'412681', '6293378'})
        assert(set(child[2].root for child in rule1.children) == {('7241965',), ('7490702',)})
        assert(len(rule1.correct_assertions) == 1)
        assert(rule1.get_edges_covered() == original_covered_edges.union(rule2.get_edges_covered()))
        assert(rule1.get_labels_covered() == original_covered_labels.union(rule2.get_labels_covered()))
        assert(len(rule1.correct_assertions[0].nodes['7499850'].neighbors_of_type) == 2)

        # try more merges
        assert(rule1.merge(rule3))
        assert(len(rule1.children) == 3)
        assert(set(child[0] for child in rule1.children) == {'412681', '6293378', '3320538'})
        assert(set(child[2].root for child in rule1.children) == {('7241965',), ('7490702',), ('8226812',)})
        assert(len(rule1.correct_assertions) == 1)
        assert(rule1.get_edges_covered() == original_covered_edges.union(rule2.get_edges_covered()).union(rule3.get_edges_covered()))
        assert(rule1.get_labels_covered() == original_covered_labels.union(rule2.get_labels_covered()).union(rule3.get_labels_covered()))
        assert(len(rule1.correct_assertions[0].nodes['7499850'].neighbors_of_type) == 3)

    def test_jaccard_sim_1(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2= Rule(('7241965',), (('412681', 'out', (('7490702',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        assert(rule1.jaccard_sim(rule2) == 3 / 6)

    def test_jaccard_sim_2(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('3320538', 'out', (('8226812',), ())),))
        rule3 = Rule(('8226812',), (('6291253', 'in', (('1927286',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        model.plant_forest(rule3)

        rule1.pin_to_leaf(rule2)
        assert(rule1.jaccard_sim(rule3) == 1 / 8)

    def test_label_children_2(self):
        '''
        Test that label_children dictionary is correct after composing rules.
        '''
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('3320538', 'out', (('8226812',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('3320538', 'out', (('8226812',), ())),))
        rule3 = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        model.plant_forest(rule3)

        for ca in rule1.correct_assertions:
            assert(set(ca.nodes[ca.root].neighbors_of_type.keys()) == {('6293378', 'out', ('7241965',))})
        for ca in rule2.correct_assertions:
            assert(set(ca.nodes[ca.root].neighbors_of_type.keys()) == {('3320538', 'out', ('8226812',))})
        for ca in rule3.correct_assertions:
            assert(set(ca.nodes[ca.root].neighbors_of_type.keys()) == {('5835005', 'out', ('5794125',))})

        rule1.pin_to_leaf(rule2)
        for ca in rule1.correct_assertions:
            assert(set(ca.nodes[ca.root].neighbors_of_type.keys()) == {('6293378', 'out', ('7241965',))})
            for child in ca.nodes[ca.root].neighbors:
                if child in {'36240', '2415820'}:
                    assert(set(ca.nodes[child].neighbors_of_type.keys()) == {('6293378', 'in', ('1927286',)), ('3320538', 'out', ('8226812',))})
                else:
                    assert(set(ca.nodes[child].neighbors_of_type.keys()) == {('6293378', 'in', ('1927286',))})

        rule1.pin_to_leaf(rule3)
        for ca in rule1.correct_assertions:
            assert(set(ca.nodes[ca.root].neighbors_of_type.keys()) == {('6293378', 'out', ('7241965',))})
            for child in ca.nodes[ca.root].neighbors:
                if child in {'36240', '2415820'}:
                    assert(set(ca.nodes[child].neighbors_of_type.keys()) == {('6293378', 'in', ('1927286',)), ('3320538', 'out', ('8226812',))})
                elif child in {'6555563', '6341376'}:
                    assert(set(ca.nodes[child].neighbors_of_type.keys()) == {('6293378', 'in', ('1927286',)), ('5835005', 'out', ('5794125',))})
                else:
                    assert(set(ca.nodes[child].neighbors_of_type.keys()) == {('6293378', 'in', ('1927286',))})

    def test_label_children_3(self):
        '''
        Test that label_children dictionary is correct after composing rules with overlap.
        '''
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('7051738', 'out', (('3029686',), ())),)))
        model.add_rule((('7241965',), (('5835005', 'out', (('5794125',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('7051738', 'out', (('3029686',), ())),))
        rule3 = Rule(('7241965',), (('5835005', 'out', (('5794125',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        model.plant_forest(rule3)

        for ca in rule1.correct_assertions:
            assert(set(ca.nodes[ca.root].neighbors_of_type.keys()) == {('6293378', 'out', ('7241965',))})
        for ca in rule2.correct_assertions:
            assert(set(ca.nodes[ca.root].neighbors_of_type.keys()) == {('7051738', 'out', ('3029686',))})
        for ca in rule3.correct_assertions:
            assert(set(ca.nodes[ca.root].neighbors_of_type.keys()) == {('5835005', 'out', ('5794125',))})

        rule1.pin_to_leaf(rule2)
        for ca in rule1.correct_assertions:
            assert(set(ca.nodes[ca.root].neighbors_of_type.keys()) == {('6293378', 'out', ('7241965',))})
            for child in ca.nodes[ca.root].neighbors:
                if child in {'879961', '6341376'}:
                    assert(set(ca.nodes[child].neighbors_of_type.keys()) == {('6293378', 'in', ('1927286',)), ('7051738', 'out', ('3029686',))})
                else:
                    assert(set(ca.nodes[child].neighbors_of_type.keys()) == {('6293378', 'in', ('1927286',))})

        rule1.pin_to_leaf(rule3)
        for ca in rule1.correct_assertions:
            assert(set(ca.nodes[ca.root].neighbors_of_type.keys()) == {('6293378', 'out', ('7241965',))})
            for child in ca.nodes[ca.root].neighbors:
                if child == '6341376':
                    assert(set(ca.nodes[child].neighbors_of_type.keys()) == {('6293378', 'in', ('1927286',)), ('7051738', 'out', ('3029686',)), ('5835005', 'out', ('5794125',))})
                elif child == '879961':
                    assert(set(ca.nodes[child].neighbors_of_type.keys()) == {('6293378', 'in', ('1927286',)), ('7051738', 'out', ('3029686',))})
                elif child == '6555563':
                    assert(set(ca.nodes[child].neighbors_of_type.keys()) == {('6293378', 'in', ('1927286',)), ('5835005', 'out', ('5794125',))})
                else:
                    assert(set(ca.nodes[child].neighbors_of_type.keys()) == {('6293378', 'in', ('1927286',))})

    def test_compose_1(self):
        rule1 = Rule(('sport',), (('team', 'out', (('team',), ())),))
        rule1b = Rule(('sport',), (('plays', 'in', (('team',), ())),))
        rule1c = Rule(('team',), (('playsAgainst', 'out', (('team',), ())),))
        rule2 = Rule(('date',), (('dateOf', 'out', (('sport',), ())),))
        rule2b = Rule(('date',), (('atDate', 'in', (('sport',), ())),))
        rule3 = Rule(('shape',), (('atDate', 'out', (('date',), ())),))
        rule3b = Rule(('shape',), (('dateOf', 'in', (('date',), ())),))
        rule1.merge(rule1b)
        rule1.pin_to_leaf(rule1c)
        rule2.merge(rule2b)
        rule3.merge(rule3b)

        rule2.pin_to_leaf(rule1)
        rule3.pin_to_leaf(rule2)
        assert(rule3.max_depth() == 4)

    def test_compose_2(self):
        rule1 = Rule(('sport',), (('has_team', 'out', (('team',), ())),))
        rule2 = Rule(('team',), (('plays', 'out', (('sport',), ())),))
        rule3 = Rule(('sport',), (('has_player', 'out', (('athlete',), ())),))

        assert(rule1.get_matching_nodes(('sport',)) == {rule1})
        rule1.pin_to_leaf(rule2)
        assert(rule1.get_matching_nodes(('sport',)) == {rule1, rule2.children[0][2]})
        rule1.pin_to_leaf(rule3)
        # rule1.plot('test')
        assert(rule1.get_matching_nodes(('sport',)) == {rule1, rule2.children[0][2]})
        # assert(rule3.max_depth() == 4)

    def test_compose_3(self):
        rule1 = Rule(('publication',), (('controlls', 'in', (('publication',), ())),))
        rule2 = Rule(('publication',), (('controlledBy', 'out', (('publication',), ())),))
        rule3 = Rule(('publication',), (('worker', 'out', (('writer',), ())),))
        rule4 = Rule(('publication',), (('worksFor', 'in', (('writer',), ())),))

        assert(rule1.merge(rule2))
        assert(rule3.merge(rule4))

        ruleA = Rule(('publication',), (('controlls', 'in', (('publication',), ())), ('controlledBy', 'out', (('publication',), ()))))
        ruleB = Rule(('publication',), (('worker', 'out', (('writer',), ())), ('worksFor', 'in', (('writer',), ()))))
        assert(rule1.tuplify() == ruleA.tuplify())
        assert(rule3.tuplify() == ruleB.tuplify())

        assert(rule1.pin_to_leaf(rule3))
        comp = Rule(('publication',), (('controlls', 'in', (('publication',), ())), ('controlledBy', 'out', (('publication',), ()))))

    def test_get_inner_nodes_1(self):
        rule = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        assert(rule.get_inner_nodes() == {('7241965',)})

    def test_get_inner_nodes_2(self):
        rule1 = Rule(('sport',), (('has_team', 'out', (('team',), ())),))
        rule2 = Rule(('team',), (('plays', 'out', (('sport',), ())),))
        rule3 = Rule(('sport',), (('has_player', 'out', (('athlete',), ())),))

        assert(rule1.get_inner_nodes() == {('team',)})
        rule1.pin_to_leaf(rule2)
        assert(rule1.get_inner_nodes() == {('team',), ('sport',)})
        rule1.pin_to_leaf(rule3)
        assert(rule1.get_inner_nodes() == {('team',), ('sport',), ('athlete',)})

    def test_get_inner_nodes_3(self):
        rule1 = Rule(('sport',), (('has_team', 'out', (('team',), ())),))
        rule2 = Rule(('team',), (('plays', 'out', (('sport',), ())),))
        rule3 = Rule(('sport',), (('has_player', 'out', (('athlete',), ())),))
        rule4 = Rule(('sport',), (('has_coach', 'out', (('coach',), ())),))

        assert(rule1.get_inner_nodes() == {('team',)})
        assert(rule1.pin_to_leaf(rule2))
        assert(rule1.get_inner_nodes() == {('team',), ('sport',)})
        assert(rule1.pin_to_leaf(rule3))
        assert(rule1.get_inner_nodes() == {('team',), ('sport',), ('athlete',)})
        assert(rule1.pin_to_leaf(rule4))
        assert(rule1.get_inner_nodes() == {('team',), ('sport',), ('athlete',), ('coach',)})

    def test_merge_1(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('1927286',), (('412681', 'out', (('7490702',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('1927286',), (('412681', 'out', (('7490702',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        assert(rule1.get_labels_covered() == {('7241965', '36240'), ('7241965', '6555563'), ('7241965', '2415820'), ('7241965', '879961'), ('7241965', '6341376'), ('7241965', '6175574')})
        assert(rule2.get_labels_covered() == {('7490702', '8220493')})
        assert(rule1.merge(rule2))
        assert(rule1.get_labels_covered() == {('7241965', '36240'), ('7241965', '6555563'), ('7241965', '2415820'), ('7241965', '879961'), ('7241965', '6341376'), ('7241965', '6175574'), ('7490702', '8220493')})

    def test_merge_2(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('1927286',), (('412681', 'out', (('7490702',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('1927286',), (('412681', 'out', (('7490702',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        assert(rule1.get_labels_covered() == {('7241965', '36240'), ('7241965', '6555563'), ('7241965', '2415820'), ('7241965', '879961'), ('7241965', '6341376'), ('7241965', '6175574')})
        assert(rule2.get_labels_covered() == {('7490702', '8220493')})
        assert(rule2.merge(rule1))
        assert(rule2.get_labels_covered() == {('7241965', '36240'), ('7241965', '6555563'), ('7241965', '2415820'), ('7241965', '879961'), ('7241965', '6341376'), ('7241965', '6175574'), ('7490702', '8220493')})

    def test_merge_3(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('1927286',), (('412681', 'out', (('7490702',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('1927286',), (('412681', 'out', (('7490702',), ())),))
        model.plant_forest(rule1)
        model.plant_forest(rule2)
        assert(rule1.get_labels_covered() == {('7241965', '36240'), ('7241965', '6555563'), ('7241965', '2415820'), ('7241965', '879961'), ('7241965', '6341376'), ('7241965', '6175574')})
        assert(rule2.get_labels_covered() == {('7490702', '8220493')})
        merged = deepcopy(rule1)
        assert(merged.merge(rule2))
        assert(merged.get_labels_covered() == {('7241965', '36240'), ('7241965', '6555563'), ('7241965', '2415820'), ('7241965', '879961'), ('7241965', '6341376'), ('7241965', '6175574'), ('7490702', '8220493')})

    def test_merge_4(self):
        '''
        Test that nodes dictionary is correct after merging rules.
        '''
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
        merged = deepcopy(rule1)
        assert(merged.merge(rule2))
        assert(merged.merge(rule3))
        assert(len(merged.correct_assertions) == 1)
        ca = merged.correct_assertions[0]
        assert(ca.root == '7499850')
        assert(set(ca.nodes.keys()) == {'7499850', '36240', '6175574', '6555563', '879961', '6341376', '2415820', '9054900', '8220493'})
        assert(ca.nodes['7499850'].neighbors == {'36240', '6175574', '6555563', '879961', '6341376', '2415820', '9054900', '8220493'})
        assert(ca.nodes['7499850'].neighbors_of_type['6293378', 'out', ('7241965',)] == {'36240', '6175574', '6555563', '879961', '6341376', '2415820'})
        assert(ca.nodes['7499850'].neighbors_of_type['412681', 'out', ('7490702',)] == {'8220493'})
        assert(ca.nodes['7499850'].neighbors_of_type['3320538', 'out', ('8226812',)] == {'9054900'})

    def test_repeated_1(self):
        graph = Graph('repeated', verbose=False)
        r1 = Rule(('a',), (('e1', 'out', (('b',), ())),))
        r2 = Rule(('a',), (('e1\'', 'in', (('b',), ())),))
        r3 = Rule(('b',), (('e2', 'out', (('c',), ())),))
        r4 = Rule(('b',), (('e2\'', 'in', (('c',), ())),))
        assert(r1.tuplify() in graph.candidates)
        assert(r2.tuplify() in graph.candidates)
        assert(r3.tuplify() in graph.candidates)
        assert(r4.tuplify() in graph.candidates)
        model = Model(graph)
        model.plant_forest(r1)
        model.plant_forest(r2)
        model.plant_forest(r3)
        model.plant_forest(r4)

        assert(len(r1.correct_assertions) == 2)
        assert(len(r2.correct_assertions) == 2)
        assert(len(r3.correct_assertions) == 2)
        assert(len(r4.correct_assertions) == 2)

        m1 = deepcopy(r1)
        assert(m1.merge(r2))
        assert(len(m1.correct_assertions) == 2)

        # first ca
        ca = m1.correct_assertions[0]
        assert(ca.root == '1')
        assert(len(ca.nodes['1'].neighbors_of_type) == 2)
        assert(len(ca.nodes['1'].neighbors_of_type[('e1', 'out', ('b',))]) == 1)
        for it in ca.nodes['1'].neighbors_of_type[('e1', 'out', ('b',))]:
            assert(it == '2')
        assert(('e1\'', 'in', ('b',)) in ca.nodes['1'].neighbors_of_type)
        for it in ca.nodes['1'].neighbors_of_type[('e1\'', 'in', ('b',))]:
            assert(it == '2')
        # second ca
        ca = m1.correct_assertions[1]
        assert(ca.root == '4')
        assert(len(ca.nodes['4'].neighbors_of_type) == 2)
        assert(('e1', 'out', ('b',)) in ca.nodes['4'].neighbors_of_type)
        assert(('e1\'', 'in', ('b',)) in ca.nodes['4'].neighbors_of_type)

        m2 = deepcopy(r3)
        assert(m2.merge(r4))
        assert(len(m2.correct_assertions) == 2)

        # first ca
        ca = m2.correct_assertions[0]
        assert(ca.root == '2')
        assert(len(ca.nodes['2'].neighbors_of_type) == 2)
        assert(('e2', 'out', ('c',)) in ca.nodes['2'].neighbors_of_type)
        assert(('e2\'', 'in', ('c',)) in ca.nodes['2'].neighbors_of_type)
        # second ca
        ca = m2.correct_assertions[1]
        assert(ca.root == '5')
        assert(len(ca.nodes['5'].neighbors_of_type) == 2)
        assert(('e2', 'out', ('c',)) in ca.nodes['5'].neighbors_of_type)
        assert(('e2\'', 'in', ('c',)) in ca.nodes['5'].neighbors_of_type)

        composed = deepcopy(m1)
        assert(composed.pin_to_leaf(m2))
        assert(len(composed.correct_assertions) == 2)

        # rist ca
        ca = composed.correct_assertions[0]
        assert(ca.root == '1')
        assert(len(ca.nodes['1'].neighbors_of_type) == 2)
        assert(len(ca.nodes['1'].neighbors_of_type[('e1', 'out', ('b',))]) == 1)
        for child in ca.nodes['1'].neighbors_of_type[('e1', 'out', ('b',))]:
            assert(child == '2')
            assert(len(ca.nodes['2'].neighbors_of_type) == 4)
            for grand_child in ca.nodes['2'].neighbors_of_type[('e2', 'out', ('c',))]:
                assert(grand_child == '3')
            for grand_child in ca.nodes['2'].neighbors_of_type[('e2\'', 'in', ('c',))]:
                assert(grand_child == '3')
        assert(len(ca.nodes['1'].neighbors_of_type[('e1\'', 'in', ('b',))]) == 1)
        for child in ca.nodes['1'].neighbors_of_type[('e1\'', 'in', ('b',))]:
            assert(child == '2')
            assert(len(ca.nodes['2'].neighbors_of_type) == 4)
            for grand_child in ca.nodes['2'].neighbors_of_type[('e2', 'out', ('c',))]:
                assert(grand_child == '3')
            for grand_child in ca.nodes['2'].neighbors_of_type[('e2\'', 'in', ('c',))]:
                assert(grand_child == '3')

    def test_get_preds_1(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('1927286',), (('412681', 'out', (('7490702',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('1927286',), (('412681', 'out', (('7490702',), ())),))
        assert(rule1.get_preds() == {'6293378'})
        assert(rule2.get_preds() == {'412681'})
        rule1.merge(rule2)
        assert(rule1.get_preds() == {'6293378', '412681'})

    def test_get_preds_2(self):
        graph = Graph('test', verbose=False)
        model = Model(graph)
        model.add_rule((('1927286',), (('6293378', 'out', (('7241965',), ())),)))
        model.add_rule((('7241965',), (('412681', 'out', (('7490702',), ())),)))
        rule1 = Rule(('1927286',), (('6293378', 'out', (('7241965',), ())),))
        rule2 = Rule(('7241965',), (('412681', 'out', (('7490702',), ())),))
        assert(rule1.get_preds() == {'6293378'})
        assert(rule2.get_preds() == {'412681'})
        rule1.pin_to_leaf(rule2)
        assert(rule1.get_preds() == {'6293378', '412681'})

if __name__ == "__main__":
    unittest.main()
