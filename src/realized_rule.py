from collections import defaultdict

class Node:
    def __init__(self, name, typ):
        self.name = name
        self.type = typ
        # maps (pred, dir, node_type) to node names
        self.neighbors_of_type = dict()
        # node names
        self.neighbors = set()

    def add_neighbor(self, pred, dir, node_type, node):
        # print('adding to {}: neighbor ({}, {}, {}, {})'.format(self.name, pred, dir, node_type, node))
        if (pred, dir, node_type) not in self.neighbors_of_type:
            self.neighbors_of_type[(pred, dir, node_type)] = set()
        self.neighbors_of_type[(pred, dir, node_type)].add(node)
        self.neighbors.add(node)

class RealizedRule:
    '''
    A class which represents realized rules as rooted subgraphs.
    '''
    def __init__(self, root, label):
        self.root = root
        # maps node names to node objects
        self.nodes = dict()
        self.edges = set()
        self.edge_ids = set()
        self.labels = set()

    def add_edge(self, edge, eid=None, labels=False):
        '''
        :edge: (u, u_typ, pred, dir, v, v_typ)
        :eid: the id of the edge in the full graph
        :labels: boolean specifying whether or not to add node labels
        '''
        u, u_typ, pred, dir, v, v_typ = edge
        sub = u if dir == 'out' else v
        sub_typ = u_typ if dir == 'out' else v_typ
        obj = u if dir == 'in' else v
        obj_typ = u_typ if dir == 'in' else v_typ
        if u not in self.nodes:
            self.nodes[u] = Node(u, u_typ)
        # print('edge: ({}, {}, {}, {}, {}, {})'.format(u, u_typ, pred, dir, v, v_typ))
        # print('sub: {} obj: {}'.format(sub, obj))
        if v not in self.nodes:
            self.nodes[v] = Node(v, v_typ)
        self.nodes[sub].add_neighbor(pred, 'out', obj_typ, obj)
        self.nodes[obj].add_neighbor(pred, 'in', sub_typ, sub)
        self.edges.add(edge)
        # edge coverage
        if eid != None:
            self.edge_ids.add(eid)
        # label coverage
        if labels:
            if u != self.root:
                for label in u_typ:
                    self.labels.add((label, u))
            if v != self.root:
                for label in v_typ:
                    self.labels.add((label, v))


    def merge(self, realized_rule):
        assert(self.root == realized_rule.root)
        for edge in realized_rule.edges:
            self.add_edge(edge)
        self.edge_ids.update(realized_rule.edge_ids)
        self.labels.update(realized_rule.labels)

    def compose(self, realized_rule):
        for edge in realized_rule.edges:
            self.add_edge(edge)
        self.edge_ids.update(realized_rule.edge_ids)
        self.labels.update(realized_rule.labels)
