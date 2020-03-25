import unittest
import sys
sys.path.append('../src/')
from graph import Graph

class TestGraph(unittest.TestCase):

    def test_load(self):
        graph = Graph('test', verbose=False)
        assert(graph.n == 18)
        assert(graph.m == 23)
        assert(graph.num_edge_labels == 7)
        assert(graph.num_node_labels == 11)
        assert(graph.edge_label_counts['6293378'] == 7)
        assert(graph.node_label_counts['7241965'] == 6)
        assert(graph.total_num_labels == 27)

    def test_load_idify(self):
        graph = Graph('test', idify=True, verbose=False)
        assert(graph.n == 18)
        assert(graph.m == 23)
        assert(graph.num_edge_labels == 7)
        assert(graph.num_node_labels == 11)
        assert(len(graph.node_to_id) == len(graph.id_to_node) == 18)
        assert(len(graph.pred_to_id) == len(graph.id_to_pred) == 7)
        assert(type(list(graph.node_label_counts.keys())[0]) is int)
        assert(type(list(graph.edge_label_counts.keys())[0]) is int)
        assert(type(graph.nodes()[0]) is int)

    def test_matches_1(self):
        graph = Graph('test', verbose=False)
        rule = (('1927286',), (('6293378', 'out', (('7241965',), ())),))
        # rule should be a candidate
        assert(rule in graph.candidates)
        # rule should have one match
        assert(len(graph.candidates[rule]['ca_to_size']) == 1)
        # edges and labels should look like this:
        for node in ['36240', '6175574', '2415820', '6341376', '6555563', '879961']:
            assert(('7241965', node) in graph.candidates[rule]['label_coverage'])
        assert(len(graph.candidates[rule]['edges']) == 6)
        assert(len(graph.candidates[rule]['label_coverage']) == 6)
        assert(sum(graph.candidates[rule]['ca_to_size'].values()) == 6)

    def test_matches_2(self):
        graph = Graph('test', verbose=False)
        rule = (('7241965',), (('7051738', 'out', (('3029686',), ())),))
        # rule should be a candidate
        assert(rule in graph.candidates)
        # rule should have two matches
        assert(len(graph.candidates[rule]['ca_to_size']) == 2)
        # edges and labels should look like this:
        assert(('3029686', '287927') in graph.candidates[rule]['label_coverage'])
        assert(sum(graph.candidates[rule]['ca_to_size'].values()) == 2)

    def test_matches_4(self):
        graph = Graph('test', verbose=False)
        rule = (('7490702',), (('412681', 'in', (('7241965',), ())),))
        # rule should be a candidate
        assert(rule in graph.candidates)
        # rule should have three matches
        assert(len(graph.candidates[rule]['ca_to_size']) == 3)
        assert(('7241965', '36240') in graph.candidates[rule]['label_coverage'])
        assert(('7241965', '6175574') in graph.candidates[rule]['label_coverage'])
        assert(('7241965', '6555563') in graph.candidates[rule]['label_coverage'])

    def test_nodes(self):
        graph = Graph('test', verbose=False)
        assert(type(graph.nodes()) is list)
        assert(len(graph.nodes()) == 18)
        nodes = ['9841316',
                 '36240',
                 '8220493',
                 '7499850',
                 '6175574',
                 '287927',
                 '9054900',
                 '2873925',
                 '919756',
                 '2415820',
                 '6341376',
                 '2211914',
                 '308389',
                 '7992351',
                 '3352101',
                 '6565312',
                 '6555563',
                 '879961']
        assert(sorted(graph.nodes()) == sorted(nodes))

    def test_labels(self):
        graph = Graph('test', verbose=False)
        assert(graph.labels('36240') == ('7241965', '6843923',))

    def test_nodes_with_type(self):
        graph = Graph('test', verbose=False)
        assert(graph.nodes_with_type(('1927286',)) == 1)
        assert(graph.nodes_with_type(('7241965',)) == 6)
        assert(graph.nodes_with_type(('8226812',)) == 3)
        assert(graph.nodes_with_type(('8226812', '5266930')) == 1)

if __name__ == "__main__":
    unittest.main()
