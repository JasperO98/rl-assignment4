from hexgame import HexGame
from hexplayer import HexPlayerHuman, HexPlayerEnhanced

if __name__ == '__main__':
    # play a game
    game = HexGame(4, HexPlayerEnhanced(10), HexPlayerHuman())
    game.play()
