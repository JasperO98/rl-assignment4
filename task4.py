from hexgame import HexGame
from hexplayer import HexPlayerHuman, HexPlayerDijkstra
from hexboard import HexBoard
from hexcolour import HexColour

if __name__ == '__main__':
    # test functionality
    board = HexBoard(3)
    board.board[(1, 1)] = HexColour.RED
    board.dijkstra(HexColour.RED, True)

    # play a game
    game = HexGame(4, HexPlayerDijkstra(4), HexPlayerHuman())
    game.play()
