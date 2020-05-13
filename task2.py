import setup
from hex.game import HexGame
from hex.players.base import HexPlayerHuman
from hex.players.selfplay import AlphaZeroSelfPlay1
from hex.board import HexBoard
from hex.alphazero import AlphaHexGame
import numpy as np
from hex.colour import HexColour


def render(matrix):
    board = HexBoard(5)
    board.set_colour(np.argwhere(matrix[:, :, 0] == 1), HexColour.RED)
    board.set_colour(np.argwhere(matrix[:, :, 0] == -1), HexColour.BLUE)
    board.render(0)


if __name__ == '__main__':
    # test AlphaZeroGeneral interactions
    game = AlphaHexGame(5, None)
    matrix = np.zeros((5, 5, 1))
    matrix[(0, 1, 2, 2, 3, 4), (1, 1, 1, 2, 2, 2), 0] = 1
    matrix[(0, 4), (4, 4), 0] = -1
    render(matrix)

    # test canonical form
    matrix = game.getCanonicalForm(matrix, -1)
    render(matrix)
    matrix = game.getCanonicalForm(matrix, -1)
    render(matrix)

    # test symmetries
    matrix = game.getSymmetries(matrix, [])[1][0]
    render(matrix)
    matrix = game.getSymmetries(matrix, [])[1][0]
    render(matrix)

    # test game ended
    print(game.getGameEnded(matrix, 1), game.getGameEnded(matrix, -1))

    # test doing moves
    matrix = game.getNextState(matrix, -1, 22)[0]
    render(matrix)

    # train and play against AlphaZeroGeneral
    player = AlphaZeroSelfPlay1(epochs=10, cp=5, episodes=50, threshold=0.5)
    HexGame(5, player, HexPlayerHuman()).play()
    HexGame(5, HexPlayerHuman(), player).play()
