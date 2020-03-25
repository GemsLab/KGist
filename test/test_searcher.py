import unittest
import sys
sys.path.append('../src/')
from graph import Graph
from searcher import Searcher

class TestSearcher(unittest.TestCase):

    def test_generate_candidates(self):
        searcher = Searcher(Graph('test', verbose=False))

        gt = {(('1927286',), (('6293378', 'out', (('766386',), ())),)),
              (('1927286',), (('6293378', 'out', (('7241965',), ())),)),
              (('1927286',), (('6293378', 'out', (('6843923',), ())),)),
              (('1927286',), (('3320538', 'out', (('5266930',), ())),)),
              (('1927286',), (('3320538', 'out', (('8226812',), ())),)),
              (('1927286',), (('6291253', 'out', (('8226812',), ())),)),
              (('1927286',), (('412681', 'out', (('7490702',), ())),)),
              (('7241965',), (('6293378', 'in', (('1927286',), ())),)),
              (('7241965',), (('5835005', 'out', (('5794125',), ())),)),
              (('7241965',), (('412681', 'out', (('7490702',), ())),)),
              (('7241965',), (('7051738', 'out', (('3029686',), ())),)),
              (('7241965',), (('7051738', 'out', (('6057655',), ())),)),
              (('7241965',), (('3320538', 'out', (('8226812',), ())),)),
              (('7241965',), (('7946920', 'out', (('8359357',), ())),)),
              (('7241965',), (('6291253', 'out', (('8226812',), ())),)),
              (('6843923',), (('6293378', 'in', (('1927286',), ())),)),
              (('6843923',), (('5835005', 'out', (('5794125',), ())),)),
              (('6843923',), (('412681', 'out', (('7490702',), ())),)),
              (('6843923',), (('7051738', 'out', (('3029686',), ())),)),
              (('6843923',), (('7051738', 'out', (('6057655',), ())),)),
              (('6843923',), (('3320538', 'out', (('8226812',), ())),)),
              (('6843923',), (('7946920', 'out', (('8359357',), ())),)),
              (('6843923',), (('6291253', 'out', (('8226812',), ())),)),
              (('5794125',), (('5835005', 'in', (('7241965',), ())),)),
              (('5794125',), (('5835005', 'in', (('6843923',), ())),)),
              (('8359357',), (('7946920', 'in', (('7241965',), ())),)),
              (('8359357',), (('7946920', 'in', (('6843923',), ())),)),
              (('5266930',), (('3320538', 'in', (('1927286',), ())),)),
              (('8226812',), (('3320538', 'in', (('1927286',), ())),)),
              (('8226812',), (('3320538', 'in', (('7241965',), ())),)),
              (('8226812',), (('3320538', 'in', (('6843923',), ())),)),
              (('7490702',), (('412681', 'in', (('1927286',), ())),)),
              (('7490702',), (('412681', 'in', (('7241965',), ())),)),
              (('7490702',), (('412681', 'in', (('6843923',), ())),)),
              (('3029686',), (('7051738', 'in', (('7241965',), ())),)),
              (('3029686',), (('7051738', 'in', (('6843923',), ())),)),
              (('6057655',), (('7051738', 'in', (('7241965',), ())),)),
              (('6057655',), (('7051738', 'in', (('6843923',), ())),)),
              (('766386',), (('6293378', 'in', (('1927286',), ())),)),
              (('8226812',), (('6291253', 'in', (('1927286',), ())),)),
              (('8226812',), (('6291253', 'in', (('7241965',), ())),)),
              (('8226812',), (('6291253', 'in', (('6843923',), ())),))}

        assert(gt == set(searcher.candidates))

    def test_build_model_1(self):
        graph = Graph('test', verbose=False)
        searcher = Searcher(graph)
        m_star = searcher.build_model(verbosity=0)

    def test_top_BoundedMinHeap_freq(self):
        graph = Graph('test', verbose=False)
        searcher = Searcher(graph)

        heap = searcher.BoundedMinHeap(bound=10, key=lambda rule: len(graph.candidates[rule]['ca_to_size']))
        for rule in searcher.candidates:
            heap.push(rule)
        rules = heap.get_reversed()
        for i in range(len(rules) - 1):
            assert(len(graph.candidates[rules[i]]['ca_to_size']) >= len(graph.candidates[rules[i + 1]]['ca_to_size']))

        heap = searcher.BoundedMinHeap(bound=5, key=lambda rule: len(graph.candidates[rule]['ca_to_size']))
        for rule in searcher.candidates:
            heap.push(rule)
        rules = heap.get_reversed()
        for i in range(len(rules) - 1):
            assert(len(graph.candidates[rules[i]]['ca_to_size']) >= len(graph.candidates[rules[i + 1]]['ca_to_size']))

        heap = searcher.BoundedMinHeap(bound=15, key=lambda rule: len(graph.candidates[rule]['ca_to_size']))
        for rule in searcher.candidates:
            heap.push(rule)
        rules = heap.get_reversed()
        for i in range(len(rules) - 1):
            assert(len(graph.candidates[rules[i]]['ca_to_size']) >= len(graph.candidates[rules[i + 1]]['ca_to_size']))
