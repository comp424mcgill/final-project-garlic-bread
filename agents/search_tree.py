# Never mind this...
class Tree:
    def __init__(self):
        self.root = None

    class Node:
        def __init__(self, game_state):
            self.children = []
            self.data = game_state
