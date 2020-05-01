from copy import deepcopy

class Rule:
    '''
    A class which represents rules as trees.
    Best used for rules after level 1.
    '''
    def __init__(self, root, children=None):
        self.root = root
        self.children = list()
        # a list of trees that match the rule structure all the way down
        #   - if empty, then the rule is not a root
        self.correct_assertions = list()
        if children:
            for child in children:
                self.insert(child)
        # store the tuple form or rules for easy (ehh) debugging. Yay!
        self.tuple_rules = [self.tuplify()]
        self.correct_at_depth = dict()
        self.assertions_at_depth = dict()

    def has_children(self):
        return len(self.children) > 0

    def instantiated(self):
        return len(self.correct_assertions) > 0

    def insert(self, branch):
        '''
        Insert a branch into the rule.

        :branch: (pred, dir, leaf)
            - :pred: the edge type
            - :dir: the direction of the edge
            - :leaf: the node id
        '''
        if type(branch[2]) is tuple:
            pred, dir, leaf = branch
            leaf, _ = leaf
            leaf = Rule(leaf)
        else:
            pred, dir, leaf = branch
        self.children.append((pred, dir, leaf))

    def insert_correct_assertion(self, correct_assertion):
        '''
        Adds a correct_assertion, which should only be used for roots.

        :correct_assertion: a CorrectAssertion
        '''
        self.correct_assertions.append(correct_assertion)

    def get_edges_covered(self):
        covered = set()
        for real in self.correct_assertions:
            covered.update(real.edge_ids)
        return covered

    def get_labels_covered(self):
        covered = set()
        for real in self.correct_assertions:
            covered.update(real.labels)
        return covered

    def merge(self, other_rule):
        '''
        Merge two rules with the same roots. For now we assume the correct assertions are identical, which is true a shocking amount of the time.
        '''
        assert(other_rule.root == self.root)

        self.children.extend(other_rule.children)
        self.correct_assertions = sorted(self.correct_assertions, key=lambda real: real.root)
        other_rule.correct_assertions = sorted(other_rule.correct_assertions, key=lambda real: real.root)
        for real, other_real in zip(self.correct_assertions, other_rule.correct_assertions):
            real.merge(other_real)
        return True

    def pin_to_leaf(self, leaf):
        '''
        Pins a Rule to a leaf, and does likewise for the rule correct_assertions.

        :leaf: a Rule object whose root is the label of the leaf to which we should pin it.

        :return: True if leaf successfully pinned, False otherwise
        '''
        found = False
        # search children for a matching leaf
        for i, child in enumerate(self.children):
            pred, dir, child = child
            if child.root == leaf.root: # pin here
                if self.instantiated():
                    for leaf_real in leaf.correct_assertions: # append each leaf correct_assertion
                        for real in self.correct_assertions: # pin to each correct_assertion as well, if appropriate
                            if leaf_real.root in real.nodes:
                                real.compose(leaf_real)
                self.children[i][2].children.extend(leaf.children)
                found = True
        if not found: # try children recursively
            for child in self.children:
                found = child[2].pin_to_leaf(leaf) or found
            return found
        else:
            return found

    def get_matching_nodes(self, label):
        matches = set()
        frontier = [self]
        while len(frontier) > 0:
            node = frontier.pop()
            if node.root == label:
                matches.add(node)
            for _, _, child in node.children:
                frontier.insert(0, child)
        return matches

    def get_leaves(self):
        leaves = list()
        def add_leaf(r):
            if not r.has_children():
                leaves.append(r.root)
            else:
                for edge_type, dir, child in r.children:
                    add_leaf(child)
        add_leaf(self)
        return leaves

    def get_inner_nodes(self):
        inner_nodes = set()
        for _, _, child in self.children:
            inner_nodes.add(child.root)
            inner_nodes.update(child.get_inner_nodes())
        return inner_nodes

    def jaccard_sim(self, other_rule):
        leaves = set()
        for real in self.correct_assertions:
            leaves.update(set(real.nodes.keys()).difference({real.root}))
        roots = set(real.root for real in other_rule.correct_assertions)
        return len(leaves.intersection(roots)) / len(leaves.union(roots)) if len(leaves.union(roots)) > 0 else 0

    def tuplify(self, id_to_node=None, id_to_pred=None):
        if id_to_node and id_to_pred:
            return (tuple(id_to_node[label] for label in self.root), tuple((id_to_pred[child[0]], child[1], child[2].tuplify(id_to_node, id_to_pred)) for child in self.children))
        return (self.root, tuple((child[0], child[1], child[2].tuplify()) for child in self.children))

    def get_atoms(self):
        atoms = list()
        for child in self.children:
            atoms.append((self.root, tuple(((child[0], child[1], (child[2].root, ())),))))
            atoms.extend(child[2].get_atoms())
        return atoms

    def get_preds(self):
        preds = set()
        for pred, dir, child in self.children:
            preds.add(pred)
            preds.update(child.get_preds())
        return preds

    def max_depth(self):
        if not self.has_children():
            return 0
        return 1 + max(child[2].max_depth() for child in self.children)

    def min_depth(self):
        if not self.has_children():
            return 0
        return 1 + min(child[2].min_depth() for child in self.children)

    def get_correct(self, with_removed=False):
        '''
        Correct means that at every level, the correct_assertions align with the rule structure.

        :return: a list of relization trees that are correct all the way down.
        '''

        def is_correct(real, real_node, node):
            '''
            :real: a correct_assertion
            :node: a rule node

            :return: true if the correct_assertion is correct from this node on
            '''
            # the types of children expected of the correct_assertion
            expected_children = list((child[0], child[1], child[2].root) for child in node.children)
            correct = True # innocent until proven guilty
            for pred, dir, child in expected_children:
                if (pred, dir, child) not in real_node.neighbors_of_type:
                    correct = False
                    break
            if correct:
                # iterate over each child of the rule node
                for pred, dir, child in node.children:
                    for real_child in real_node.neighbors_of_type[(pred, dir, child.root)]:
                        correct = is_correct(real, real.nodes[real_child], child)
                        if not correct:
                            break
                    if not correct:
                        break
            return correct

        return list(filter(lambda real: is_correct(real, real.nodes[real.root], self), self.correct_assertions))

    def filter_errant(self):
        '''
        Filter out the correct_assertions that err at some level.
        '''
        # bfs
        self.correct_assertions = self.get_correct()

    def print(self):
        frontier = [(None, None, self, 0)]
        while len(frontier) > 0:
            pred, dir, rule, depth = frontier.pop()
            if depth == 0:
                print('{}{}'.format(''.join(['\t'] * depth), rule.root))
            else:
                print('{}({},{},{})'.format(''.join(['\t'] * depth), pred, dir, rule.root))
            for pred, dir, child in rule.children:
                frontier.append((pred, dir, child, depth + 1))

    def plot(self, save_path=None):
        '''
        Create a plot of a rule.
        '''
        from graphviz import Digraph

        dot = Digraph()
        root = self
        frontier = [self]
        edges = list()
        while len(frontier) > 0:
            node = frontier.pop()
            if node == root:
                dot.attr('node', shape='doublecircle')
            else:
                dot.attr('node', shape='circle')
            node_name = node.root[0].split(':')[-1]
            dot.node(node_name, node_name)
            for pred, delta, child in node.children:
                frontier.insert(0, child)
                node_name = node.root[0].split(':')[-1]
                child_name = child.root[0].split(':')[-1]
                pred = pred.split(':')[-1]
                if delta == 'out':
                    edges.append((node_name, pred, child_name))
                else:
                    edges.append((child_name, pred, node_name))

        for edge in edges:
            dot.edge(edge[0], edge[2], label=' {}'.format(edge[1]))

        save_path = save_path if save_path else '../data/temp/rule_fig'
        dot.render(save_path, view=True)

    def json_dump(self, outfile=None):
        import json
        data = {'root': self.root,
                'children': list({'pred': pred, 'dir': dir, 'rule': child.json_dump()} for pred, dir, child in self.children)}
        if outfile:
            with open(outfile, 'w') as outfile:
                    json.dump(data, outfile, indent=4)
        return data
