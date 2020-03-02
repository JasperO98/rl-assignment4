from hex.game import HexGame
from hex.players.base import HexPlayerHuman
from hex.players.montecarlo import HexPlayerMonteCarlo

if __name__ == '__main__':
    # play a game
    game = HexGame(5, HexPlayerMonteCarlo(10), HexPlayerHuman())
    game.play()
