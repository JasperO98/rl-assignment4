import setup
from hex.game import HexGame
from hex.players.base import HexPlayerHuman
from hex.players.selfplay import AlphaZeroSelfPlay1

if __name__ == '__main__':
    game = HexGame(7, AlphaZeroSelfPlay1(), HexPlayerHuman())
    game.play()
