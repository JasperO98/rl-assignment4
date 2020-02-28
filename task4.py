from hexgame import HexGame
from hexplayer import HexPlayerHuman, HexPlayerDijkstra
from hexboard import HexBoard
from hexcolour import HexColour

if __name__ == '__main__':
    # test functionality
    board = HexBoard(3)
    board.set_colour([(0, 1), (1, 1), (2, 1)], HexColour.RED)
    board.dijkstra(HexColour.RED, True)
    board.dijkstra(HexColour.BLUE, True)

    board = HexBoard(5)
    board.set_colour([(0, 3), (2, 2), (4, 1)], HexColour.RED)
    board.dijkstra(HexColour.RED, True)

    # play a game
    game = HexGame(5, HexPlayerDijkstra(4), HexPlayerHuman())
    game.play()
