from hexgame import HexGame
from hexplayer import HexPlayerHuman, HexPlayerDijkstra

if __name__ == '__main__':
    game = HexGame(4, HexPlayerDijkstra(4), HexPlayerHuman())
    game.play()
