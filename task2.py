import setup
from hex.game import HexGame
from hex.players.base import HexPlayerHuman, HexPlayerRandom
from hex.players.selfplay import AlphaZeroSelfPlay1

if __name__ == '__main__':
    # ensure trained model exists
    game = HexGame(7, AlphaZeroSelfPlay1(), HexPlayerRandom())
    game.step([])

    # play against trained model
    game = HexGame(7, AlphaZeroSelfPlay1(), HexPlayerHuman())
    game.play()
