import unittest
from test_evaluator import TestEvaluator
from test_graph import TestGraph
from test_model import TestModel
from test_searcher import TestSearcher
from test_rule import TestRule

'''
A script to run all the test cases.
'''
# load test suites
evaluator_suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluator)
graph_suite = unittest.TestLoader().loadTestsFromTestCase(TestGraph)
model_suite = unittest.TestLoader().loadTestsFromTestCase(TestModel)
searcher_suite = unittest.TestLoader().loadTestsFromTestCase(TestSearcher)
rule_suite = unittest.TestLoader().loadTestsFromTestCase(TestRule)
# combine the test suites
suites = unittest.TestSuite([graph_suite,
                             model_suite,
                             evaluator_suite,
                             searcher_suite,
                             rule_suite])
# run the test suites
unittest.TextTestRunner(verbosity=2).run(suites)
