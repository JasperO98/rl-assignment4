from hex.players.base import HexPlayer
import igraph as ig


class HexPlayerMonteCarlo(HexPlayer):
    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout
        self.tree = ig.Graph()

    def __str__(self):
        return 'MCTS\n(timeout ' + str(self.timeout) + ')'

    def get_move(self, board, colour, renders):
        pass
