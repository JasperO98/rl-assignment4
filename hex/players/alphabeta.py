import numpy as np
import igraph as ig
from func_timeout import func_timeout, FunctionTimedOut
from itertools import count
from time import time
from random import randint
from hex.players.base import HexPlayer


class HexPlayerRandomAB(HexPlayer):
    def __init__(self, depth):
        super().__init__()
        self.tree_cur = None
        self.tree_prev = None
        self.depth = depth
        self.use_tt = False

    def __str__(self):
        return 'AB Random\n(depth ' + str(self.depth) + ')'

    def eval(self, board):
        return randint(-9, 9)

    def maybe_show_tree(self, renders):
        if 'tree' in renders:
            ig.plot(
                obj=self.tree_cur,
                layout=self.tree_cur.layout_reingold_tilford(),
                margin=30,
                vertex_label_dist=-0.5,
                bbox=(1024, 512),
            )

    def determine_move(self, board, renders):
        self.tree_cur = ig.Graph(directed=True)
        alphabeta = self.alphabeta(True, self.depth, -np.inf, np.inf, board)
        self.maybe_show_tree(renders)
        return alphabeta

    def board_score_for_id(self, data):
        if self.use_tt and self.tree_prev is not None:
            try:
                return self.tree_prev.vs.find(hash=hash(data[0]))['value']
            except (ValueError, KeyError):
                pass
        return 0

    def alphabeta(self, top, depth, lower, upper, board):
        # check transposition table for board state
        if self.use_tt:
            try:
                return self.tree_cur.vs.find(hash=hash(board))
            except (ValueError, KeyError):
                pass

        # leaf node
        if depth == 0 or board.is_game_over():
            value = self.eval(board)
            return self.tree_cur.add_vertex(
                label=value,
                hash=hash(board),
                value=value,
                size=22,
            )

        # track best move and child vertices
        best = None
        vertices = []

        # iterate over child nodes
        for child, move in sorted(
                board.children(),
                key=self.board_score_for_id,
                reverse=board.turn() == self.colour,
        ):
            pass

            # get data for child node
            vertices.append((
                self.alphabeta(False, depth - 1, lower, upper, child),
                self.move_to_string(move),
            ))
            bound = vertices[-1][0]['value']

            # update global bounds
            if board.turn() == self.colour and bound > lower:
                lower = bound
                best = move
            if board.turn() == -self.colour and bound < upper:
                upper = bound
                best = move

            # stop when bounds mismatch
            if upper <= lower:
                break

        # update proof tree
        parent = self.tree_cur.add_vertex(
            label='(' + str(lower) + ',' + str(upper) + ')',
            hash=hash(board),
            value=lower if board.turn() == self.colour else upper,
            width=55,
            height=22,
            shape='rectangle',

        )
        for vertex in vertices:
            self.tree_cur.add_edge(parent, vertex[0], label=vertex[1])

        # return appropriate bound (or best move)
        if top:
            return best
        else:
            return parent


class HexPlayerDijkstraAB(HexPlayerRandomAB):
    def eval(self, board):
        return board.dijkstra(-self.colour) - board.dijkstra(self.colour)

    def __str__(self):
        return 'AB Dijkstra\n(depth ' + str(self.depth) + ')'


class HexPlayerEnhancedAB(HexPlayerDijkstraAB):
    def __init__(self, timeout, use_tt):
        super().__init__(timeout)
        self.use_tt = use_tt
        self.reached = 0

    def __str__(self):
        return 'ID' + ('TT' if self.use_tt else '') + '\n(timeout ' + str(self.depth) + 's)'

    def determine_move(self, board, renders):
        stop = time() + self.depth
        alphabeta = None

        for i in count(1):
            self.tree_prev = self.tree_cur
            self.tree_cur = ig.Graph(directed=True)
            try:
                alphabeta = func_timeout(stop - time(), self.alphabeta, (True, i, -np.inf, np.inf, board))
            except FunctionTimedOut:
                return alphabeta
            self.maybe_show_tree(renders)
            self.reached = i
