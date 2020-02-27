import re
from abc import ABC, abstractmethod
import numpy as np
from hexcolour import HexColour
import igraph as ig
from func_timeout import func_timeout, FunctionTimedOut
from itertools import count
from time import time
from copy import deepcopy


class HexPlayer(ABC):
    INSTANCE = HexColour.RED

    def __init__(self):
        self.colour = HexPlayer.INSTANCE
        HexPlayer.INSTANCE = HexPlayer.INSTANCE.invert()

    @abstractmethod
    def get_move(self, board, renders):
        pass


class HexPlayerHuman(HexPlayer):
    def get_move(self, board, renders):
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
        self.board = None
        self.tree = None
        self.depth = depth

    def eval(self):
        return randint(-9, 9)

    def get_move(self, board, renders):
        self.board = board
        self.tree = ig.Graph()
        alphabeta = self.alphabeta(True, self.depth, -np.inf, np.inf)
        if 'tree' in renders:
            ig.plot(obj=self.tree, layout=self.tree.layout_reingold_tilford())
        return alphabeta

    def alphabeta(self, top, depth, lower, upper):
        # leaf node
        if depth == 0 or self.board.is_game_over():
            value = self.eval()
            return value, self.tree.add_vertex(label=value)

        # track best move and child vertices
        best = None
        vxcs = []

        # iterate over child nodes
        for move in self.board.possible_moves():
            pass

            # get bound for child node
            self.board.do_move(move)
            bound, vxc = self.alphabeta(False, depth - 1, lower, upper)
            vxcs.append(vxc)
            self.board.undo_move()

            # update global bounds
            if self.board.turn() == self.colour and bound > lower:
                lower = bound
                best = move
            if self.board.turn() == self.colour.invert() and bound < upper:
                upper = bound
                best = move

            # stop when bounds mismatch
            if upper <= lower:
                break

        # update proof tree
        vxp = self.tree.add_vertex(label='(' + str(lower) + ',' + str(upper) + ')')
        self.tree.add_edges(((vxp, vxc) for vxc in vxcs))

        # return appropriate bound (or best move)
        if top:
            return best
        else:
            return (lower, vxp) if self.board.turn() == self.colour else (upper, vxp)


class HexPlayerDijkstra(HexPlayerRandom):
    def eval(self):
        return self.board.dijkstra(self.colour.invert()) - self.board.dijkstra(self.colour)


class HexPlayerEnhanced(HexPlayerDijkstra):
    def get_move(self, board, renders):
        self.board = deepcopy(board)
        stop = time() + self.depth
        alphabeta = None

        for i in count(1):
            self.tree = ig.Graph()
            try:
                alphabeta = func_timeout(stop - time(), self.alphabeta, (True, i, -np.inf, np.inf))
            except FunctionTimedOut:
                return alphabeta
            if 'tree' in renders:
                ig.plot(obj=self.tree, layout=self.tree.layout_reingold_tilford())
