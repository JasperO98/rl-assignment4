import setup
from hex.game import HexGame
from hex.players.base import HexPlayerHuman, HexPlayerRandom
from hex.players.selfplay import AlphaZeroSelfPlay1

if __name__ == '__main__':
    player = AlphaZeroSelfPlay1(epochs=10, cp=5)

    # ensure trained model exists
    game = HexGame(5, player, HexPlayerRandom())
    game.step([])

    # play against trained model
    game = HexGame(5, player, HexPlayerHuman())
    game.play()
