from hexgame import HexGame
from hexplayer import HexPlayerHuman, HexPlayerRandom

if __name__ == '__main__':
    game = HexGame(4, HexPlayerRandom(4), HexPlayerHuman())
    game.play()
