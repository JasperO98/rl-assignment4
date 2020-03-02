from hex.players.base import HexPlayer
import igraph as ig
import numpy.random as npr
from time import time
import numpy as np
from math import sqrt, log
from ctypes import c_long, sizeof


class HexPlayerMonteCarlo(HexPlayer):
    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout
        self.tree = ig.Graph(directed=True)

    def __str__(self):
        return 'MCTS\n(timeout ' + str(self.timeout) + ')'

    def maybe_show_tree(self, renders):
        if 'tree' in renders:
            ig.plot(
                obj=self.tree,
                layout=self.tree.layout_reingold_tilford(),
                vertex_label_dist=-0.5,
                margin=30,
                bbox=(1024, 512),
                vertex_width=55,
                vertex_height=22,
                vertex_shape='rectangle',
            )

    def uct_for_board(self, parent, board):
        try:
            child = self.tree.vs.find(hash=hash(board))
            return child['wins'] / child['visits'] + sqrt(log(parent['visits']) / child['visits'])
        except (ValueError, KeyError):
            return np.inf

    def get_move(self, board, colour, renders):
        try:
            parent = self.tree.vs.find(hash=hash(board))
            self.tree.delete_vertices(set(range(len(self.tree.vs))) - set(self.tree.neighborhood(
                vertices=parent,
                order=2 ** (sizeof(c_long) * 8 - 1) - 1,  # maximum for a C long on this system
                mode=ig.OUT,
            )))
        except (ValueError, KeyError):
            self.tree.delete_vertices(self.tree.vs)

        stop = time() + self.timeout
        while time() < stop:
            self.walk(board, colour)
            self.maybe_show_tree(renders)

        parent = self.tree.vs.find(hash=hash(board))
        child = max(npr.permutation(parent.successors()), key=lambda v: v['wins'] / v['visits'])
        return self.string_to_move(self.tree.es[self.tree.get_eid(parent, child)]['label'])

    def walk(self, board, colour):
        # get current vertex
        try:
            parent = self.tree.vs.find(hash=hash(board))
        except (ValueError, KeyError):
            parent = self.tree.add_vertex(hash=hash(board), wins=0, visits=0)

        # leaf node stuff
        if board.is_game_over():
            won = board.check_win(colour)

        # normal node stuff
        else:
            child, move = max(npr.permutation(list(board.children())), key=lambda x: self.uct_for_board(parent, x[0]))
            child, won = self.walk(child, colour)
            if child not in parent.successors():
                self.tree.add_edge(parent, child, label=self.move_to_string(move))

        # update current vertex
        if won:
            parent['wins'] += 1
        parent['visits'] += 1
        parent['label'] = str(parent['wins']) + '/' + str(parent['visits'])

        # return current vertex
        return parent, won
