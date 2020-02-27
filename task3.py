from hexgame import HexGame
from hexplayer import HexPlayerHuman, HexPlayerRandom

if __name__ == '__main__':
    # test functionality
    game = HexGame(2, HexPlayerRandom(2), None)
    game.step(['tree'])

    # play a game
    game = HexGame(5, HexPlayerRandom(4), HexPlayerHuman())
    game.play()
