from hex.game import HexGame
from hex.players.base import HexPlayerHuman
from hex.players.montecarlo import HexPlayerMonteCarlo

if __name__ == '__main__':
    # test functionality
    game = HexGame(2, HexPlayerMonteCarlo(5, 1), HexPlayerMonteCarlo(5, 1))
    game.step(['tree'])
    game.step([])
    game.step(['tree'])

    # play a game
    game = HexGame(5, HexPlayerMonteCarlo(300, 1), HexPlayerHuman())
    game.play()
