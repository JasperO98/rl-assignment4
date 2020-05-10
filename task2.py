import setup
from hex.game import HexGame
from hex.players.base import HexPlayerHuman
from hex.players.selfplay import AlphaZeroSelfPlay2

if __name__ == '__main__':
    game = HexGame(5, AlphaZeroSelfPlay2(), HexPlayerHuman())
    game.play()
