class Community:
    """A community in the graph.

    Attributes:
        id (int): the id for the community.
        nodes (list[int]): a list of node ids which are part of the community.
        size (int): the number of nodes in the community.
    """

    def __init__(self, id=None, nodes=[], size=None):
        self.id = id
        self.size = size
        self.nodes = nodes

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes
        if len(nodes) > 0:
            self.size = len(self.nodes)
