import argparse
from graph import Graph
from model import Model
from searcher import Searcher
from evaluator import Evaluator

def parse_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph',
                        '-g',
                        type=str,
                        required=True,
                        help='The name of graph: nell or dbpedia if using our parsed data.')
    parser.add_argument('--rule_merging',
                        '-Rm',
                        type=str2bool,
                        default=False,
                        required=False,
                        help='If True, then run merging refinement (Section 4.2.2).')
    parser.add_argument('--rule_nesting',
                        '-Rn',
                        type=str2bool,
                        default=False,
                        required=False,
                        help='If True, then run nesting refinement (Section 4.2.2).')
    parser.add_argument('--idify',
                        '-i',
                        type=str2bool,
                        default=True,
                        required=False,
                        help='If True, then convert entities to integer ids for faster processing.')
    parser.add_argument('--verbosity',
                        '-v',
                        type=int,
                        default=1000000,
                        required=False,
                        help='How often to print output. If 0, then silence output.')
    parser.add_argument('--output_path',
                        '-o',
                        type=str,
                        default='../output/',
                        help='path for output and log files')
    return parser.parse_args()

def main(args):
    graph = Graph(args.graph, idify=args.idify, verbose=args.verbosity > 0)
    if args.verbosity > 0:
        print('Graph loaded.')
    searcher = Searcher(graph)
    if args.verbosity > 0:
        print('Creating model.')

    model = searcher.build_model(verbosity=args.verbosity,
                                 passes=2,
                                 label_qualify=True,
                                 order=['mdl_err', 'coverage', 'lex'])
    if args.verbosity > 0:
        print('***** Initial model *****')
        model.print_stats()
        model.save('{}{}_model'.format(args.output_path, args.graph))

    if args.rule_merging:
        model = model.merge_rules(verbosity=args.verbosity)
        if args.verbosity > 0:
            print('***** Model refined with Rm *****')
            model.print_stats()
            model.save('{}{}_model_Rm'.format(args.output_path, args.graph))

    if args.rule_nesting:
        model = model.nest_rules(verbosity=args.verbosity)
        if args.verbosity:
            print('***** Model refined with Rn *****')
            model.print_stats()
            model.save('{}{}_model_Rm_Rn'.format(args.output_path, args.graph))

if __name__ == "__main__":
    args = parse_args()
    main(args)
