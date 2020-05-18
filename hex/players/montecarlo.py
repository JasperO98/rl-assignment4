from hex.players.base import HexPlayer
import numpy.random as npr
import numpy as np
from tqdm import trange
from time import time


class CacheMCTS:
    def __init__(self):
        self.data = {}

    def get(self, item):
        item = hash(item)
        if item not in self.data:
            self.data[item] = [0, 0]
        return self.data[item]


class HexPlayerMonteCarloIterations(HexPlayer):
    def __init__(self, n, cp):
        super().__init__()
        self.limit = n
        self.cp = cp
        self.cache = CacheMCTS()

    def __str__(self):
        return 'MCTS\n(N=' + str(self.limit) + ', Cₚ=' + str(self.cp) + ')'

    def determine_move(self, board, renders):
        self.monte_carlo(board, renders)
        children, moves = npr.permutation(list(board.children())).T
        data = np.array(list(map(self.cache.get, children))).T
        return moves[np.argmax(data[0] / data[1])]

    def monte_carlo(self, board, renders):
        for _ in (
                trange(self.limit) if 'progress' in renders else range(self.limit)
        ):
            self.walk(board)

    def walk(self, board):
        cached = self.cache.get(board)
        cached[1] += 1

        if board.check_win(self.colour):
            cached[0] += 1
            return True

        if board.check_win(-self.colour):
            return False

        children = npr.permutation([child[0] for child in board.children()])
        data = np.array(list(map(self.cache.get, children))).T
        uct = data[0] / data[1] + self.cp * np.sqrt(np.log(cached[1]) / data[1])
        uct[np.isnan(uct)] = np.inf
        won = self.walk(children[np.argmax(uct)])

        if won:
            cached[0] += 1
        return won


class HexPlayerMonteCarloTime(HexPlayerMonteCarloIterations):
    def __init__(self, timeout, cp):
        super().__init__(timeout, cp)

    def __str__(self):
        return 'MCTS\n(timeout=' + str(self.limit) + 's, Cₚ=' + str(self.cp) + ')'

    def monte_carlo(self, board, renders):
        stop = time() + self.limit
        while time() < stop:
            self.walk(board)
