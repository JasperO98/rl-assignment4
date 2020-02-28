import re
from abc import ABC, abstractmethod
import numpy as np
from trueskill import TrueSkill
import igraph as ig
from func_timeout import func_timeout, FunctionTimedOut
from itertools import count
from time import time
from random import randint


class HexPlayer(ABC):
    ENV = TrueSkill(mu=25, sigma=8.333)
    def __init__(self):
        self.rating = HexPlayer.ENV.create_rating() #VC
    @abstractmethod
    def get_move(self, board, colour, renders):
        pass



class HexPlayerHuman(HexPlayer):
    def get_move(self, board, colour, renders):
        while True:

            match = re.match(r'^([0-9]+)([a-z])$', input('Coordinates: ').lower())

            if not match:
                print('Invalid Move')
                continue

            row = int(match.groups()[0])
            column = ord(match.groups()[1]) - ord('a')

            if not board.exists((row, column)) or not board.is_empty((row, column)):
                print('Invalid Move')
                continue

            return row, column


class HexPlayerRandom(HexPlayer):
    def __init__(self, depth):
        super().__init__()
        self.tree_cur = None
        self.tree_prev = None
        self.depth = depth
        self.use_tt = False

    def eval(self, board, colour):
        return randint(-9, 9)

    def get_move(self, board, colour, renders):
        self.tree_cur = ig.Graph(directed=True)
        alphabeta = self.alphabeta(True, self.depth, -np.inf, np.inf, board, 'R', colour)
        if 'tree' in renders:
            ig.plot(obj=self.tree_cur, layout=self.tree_cur.layout_reingold_tilford())
        return alphabeta

    def board_score_for_id(self, data):
        if self.use_tt and self.tree_prev is not None:
            try:
                return self.tree_prev.vs.find(hash=hash(data[0]))['value']
            except (ValueError, KeyError):
                pass
        return 0

    def alphabeta(self, top, depth, lower, upper, board, label, colour):
        # check transposition table for board state
        if self.use_tt:
            try:
                return self.tree_cur.vs.find(hash=hash(board))
            except (ValueError, KeyError):
                pass

        # leaf node
        if depth == 0 or board.is_game_over():
            value = self.eval(board, colour)
            return self.tree_cur.add_vertex(
                label=label + ' (' + str(value) + ')',
                hash=hash(board),
                value=value,
            )

        # track best move and child vertices
        best = None
        vertices = []

        # iterate over child nodes
        for child, move in sorted(
                board.children(),
                key=self.board_score_for_id,
                reverse=board.turn() == colour,
        ):
            pass

            # get data for child node
            vertices.append(
                self.alphabeta(False, depth - 1, lower, upper, child, str(move[0]) + chr(move[1] + ord('a')), colour)
            )
            bound = vertices[-1]['value']

            # update global bounds
            if board.turn() == colour and bound > lower:
                lower = bound
                best = move
            if board.turn() == colour.invert() and bound < upper:
                upper = bound
                best = move

            # stop when bounds mismatch
            if upper <= lower:
                break

        # update proof tree
        parent = self.tree_cur.add_vertex(
            label=label + ' (' + str(lower) + ',' + str(upper) + ')',
            hash=hash(board),
            value=lower if board.turn() == colour else upper,
        )
        self.tree_cur.add_edges(((parent, vertex) for vertex in vertices))

        # return appropriate bound (or best move)
        if top:
            return best
        else:
            return parent


class HexPlayerDijkstra(HexPlayerRandom):
    def eval(self, board, colour):
        return board.dijkstra(colour.invert()) - board.dijkstra(colour)


class HexPlayerEnhanced(HexPlayerDijkstra):
    def __init__(self, timeout, use_tt):
        super().__init__(timeout)
        self.use_tt = use_tt

    def get_move(self, board, renders, colour):
        stop = time() + self.depth
        alphabeta = None

        for i in count(1):
            self.tree_prev = self.tree_cur
            self.tree_cur = ig.Graph(directed=True)
            try:
                alphabeta = func_timeout(stop - time(), self.alphabeta, (True, i, -np.inf, np.inf, board, 'R', colour))
            except FunctionTimedOut:
                return alphabeta
            if 'tree' in renders:
                ig.plot(obj=self.tree_cur, layout=self.tree_cur.layout_reingold_tilford())
