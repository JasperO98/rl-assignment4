from hex.board import HexBoard
from alphazero.Game import Game
import numpy as np
from hex.colour import HexColour


class AlphaHexGame(Game):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def getInitBoard(self):
        return np.zeros((self.size, self.size), int)

    def getBoardSize(self):
        return self.size, self.size

    def getActionSize(self):
        return self.size * self.size

    def getNextState(self, board, player, action):
        board = deepcopy(board)
        board[divmod(action, self.size)] = player
        return board, -player

    def getValidMoves(self, board, player):
        return board.flatten() == 0

    def getGameEnded(self, board, player):
        hex_board = HexBoard(self.size)

        hex_board.set_colour(np.argwhere(board == player), HexColour.RED)
        hex_board.set_colour(np.argwhere(board == -player), HexColour.BLUE)

        if hex_board.check_win(HexColour.RED):
            return player
        if hex_board.check_win(HexColour.BLUE):
            return -player
        return 0

    def getCanonicalForm(self, board, player):
        return board * player

    def getSymmetries(self, board, pi):
        return (
            (board, pi),
            (np.rot90(board, 2), np.flip(pi)),
        )

    def stringRepresentation(self, board):
        return board.tostring()
