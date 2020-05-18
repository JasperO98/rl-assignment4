import numpy as np
from func_timeout import func_timeout, FunctionTimedOut
from itertools import count
from time import time
from random import randint
from hex.players.base import HexPlayer


class TranspositionTable:
    def __init__(self, enabled):
        self.prev = {}
        self.cur = {}
        self.enabled = enabled

    def rotate(self):
        self.prev = self.cur
        self.cur = {}

    def set(self, board, value):
        self.cur[hash(board)] = value

    def get_for_cache(self, board):
        if not self.enabled:
            return False, None

        board = hash(board)
        if board not in self.cur:
            return False, None
        return True, self.cur[board]

    def get_for_ordering(self, child):
        if not self.enabled:
            return 0

        child = hash(child[0])
        if child not in self.prev:
            return 0
        return self.prev[child]


class HexPlayerRandomAB(HexPlayer):
    def __init__(self, depth):
        super().__init__()
        self.tt = None
        self.depth = depth
        self.use_tt = False

    def __str__(self):
        return 'AB Random\n(depth=' + str(self.depth) + ')'

    def eval(self, board):
        return randint(-9, 9)

    def determine_move(self, board, renders):
        self.tt = TranspositionTable(self.use_tt)
        return self.alphabeta(True, self.depth, -np.inf, np.inf, board)

    def alphabeta(self, top, depth, lower, upper, board):
        # check transposition table for board state
        exists, value = self.tt.get_for_cache(board)
        if exists:
            return value

        # leaf node
        if depth == 0 or board.is_game_over():
            value = self.eval(board)
            self.tt.set(board, value)
            return value

        # track best move
        best = None

        # iterate over child nodes
        for child, move in sorted(
                board.children(),
                key=self.tt.get_for_ordering,
                reverse=board.turn() == self.colour,
        ):
            pass

            # get data for child node
            bound = self.alphabeta(False, depth - 1, lower, upper, child)

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

        # update transposition table
        value = lower if board.turn() == self.colour else upper
        self.tt.set(board, value)

        # return appropriate bound (or best move)
        if top:
            return best
        else:
            return value


class HexPlayerDijkstraAB(HexPlayerRandomAB):
    def eval(self, board):
        return board.dijkstra(-self.colour) - board.dijkstra(self.colour)

    def __str__(self):
        return 'AB Dijkstra\n(depth=' + str(self.depth) + ')'


class HexPlayerEnhancedAB(HexPlayerDijkstraAB):
    def __init__(self, timeout, use_tt):
        super().__init__(timeout)
        self.use_tt = use_tt
        self.reached = 0

    def __str__(self):
        return 'ID' + ('TT' if self.use_tt else '') + '\n(timeout=' + str(self.depth) + 's)'

    def determine_move(self, board, renders):
        self.tt = TranspositionTable(self.use_tt)
        stop = time() + self.depth
        alphabeta = None

        for i in count(1):
            try:
                alphabeta = func_timeout(stop - time(), self.alphabeta, (True, i, -np.inf, np.inf, board))
            except FunctionTimedOut:
                return alphabeta
            self.reached = i
            self.tt.rotate()
