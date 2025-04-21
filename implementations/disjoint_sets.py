class DJNode:
    def __init__(self, val):
        self.val = val
        self.parent = self
        self.size = 1
        self.rank = 0

class disjoint_sets:


    def __init__(self):
        self.nodes = {}

    def insert(self, iter):
        for i in iter:
            self.nodes[i] = DJNode(i)

    def find(self, i):
        x= self.nodes[i]
        root = x
        while x.parent != root:
            root=root.parent
        while x.parent != root:
            parent = x.parent
            x.parent=root
            x=parent
        return root

    def union(self, i, j):
        x=self.find(i)
        y=self.find(j)
        if x==y:
            return
        if x.size < y.size:
            x,y = y,x
        y.parent = x
        x.size = x.size+y.size
