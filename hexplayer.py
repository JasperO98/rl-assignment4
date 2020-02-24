import re
from abc import ABC, abstractmethod
import numpy as np
from hexcolour import HexColour


class HexPlayer(ABC):
    INSTANCE = HexColour.RED

    def __init__(self):
        self.colour = HexPlayer.INSTANCE
        HexPlayer.INSTANCE = HexPlayer.INSTANCE.invert()

    @abstractmethod
    def get_move(self, board):
        pass


class HexPlayerHuman(HexPlayer):
    def get_move(self, board):
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


class HexPlayerAlphaBeta(HexPlayer):
    def __init__(self):
        super().__init__()
        self.board = None

    def get_move(self, board):
        self.board = board
        return self.alphabeta(True, 3, -np.inf, np.inf)

    def alphabeta(self, top, depth, lower, upper):
        # leaf node
        if depth == 0 or self.board.is_game_over():
            return self.board.dijkstra(self.colour) - self.board.dijkstra(self.colour.invert())

        # track best move
        best = None

        # iterate over child nodes
        for move in self.board.possible_moves():
            pass

            # get bound for child node
            self.board.do_move(move)
            bound = self.alphabeta(False, depth - 1, lower, upper)
            self.board.undo_move(move)

            # update global bounds
            if self.board.turn() and bound > lower:
                lower = bound
                best = move
            if not self.board.turn() and bound < upper:
                upper = bound
                best = move

            # stop when bounds mismatch
            if upper <= lower:
                break

        # return appropriate bound (or best move)
        if top:
            return best
        else:
            return lower if self.board.turn() else upper
