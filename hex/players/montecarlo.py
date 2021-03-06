from hex.players.base import HexPlayer
import numpy.random as npr
import numpy as np
from tqdm import trange
from time import time


class CacheMCTS:
    def __init__(self):
        self.data = {}

    def get(self, board):
        board = hash(board)
        if board not in self.data:
            self.data[board] = [0, 0]
        return self.data[board]


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
        good = data[1] != 0
        winrate = np.zeros(len(moves))
        winrate[good] = data[0, good] / data[1, good]
        return moves[np.argmax(winrate)]

    def monte_carlo(self, board, renders):
        for _ in (
                trange(self.limit) if 'progress' in renders else range(self.limit)
        ):
            self.walk(board)

    def walk(self, board):
        cached = self.cache.get(board)
        cached[1] += 1

        if len(board.board) == board.size ** 2:
            if board.check_win(self.colour):
                cached[0] += 1
                return True
            else:
                return False

        children = npr.permutation([child[0] for child in board.children()])

        if cached[1] == 1:
            won = self.walk(children[0])

        else:
            data = np.array(list(map(self.cache.get, children))).T
            good = data[1] != 0
            uct = np.ones(len(children)) * np.inf
            uct[good] = data[0, good] / data[1, good] + self.cp * np.sqrt(np.log(cached[1] - 1) / data[1, good])
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
