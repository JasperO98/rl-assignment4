import setup
from hex.game import HexGame
from hex.players.base import HexPlayerRandom
from hex.players.selfplay import AlphaZeroSelfPlay1

if __name__ == '__main__':
    HexGame(5, AlphaZeroSelfPlay1(epochs=10, cp=5), HexPlayerRandom()).play(['win'])
