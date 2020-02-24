from hexgame import HexGame
from hexplayer import HexPlayerHuman, HexPlayerAlphaBeta

if __name__ == '__main__':
    game = HexGame(4, HexPlayerAlphaBeta(), HexPlayerHuman())
    game.play()
