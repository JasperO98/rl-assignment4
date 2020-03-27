from hex.game import HexGame
from hex.players.base import HexPlayerHuman, HexPlayerRandom
from hex.players.montecarlo import HexPlayerMonteCarloIterations

if __name__ == '__main__':
    # test functionality
    game = HexGame(2, HexPlayerMonteCarloIterations(5, 1), HexPlayerRandom())
    game.step(['tree'])
    game.step(['board'])
    game.step(['tree'])

    # play a game
    game = HexGame(5, HexPlayerMonteCarloIterations(300, 1), HexPlayerHuman())
    game.play()
