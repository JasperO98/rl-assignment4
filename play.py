from hexgame import HexGame
from hexplayer import HexPlayerHuman

if __name__ == '__main__':
    game = HexGame(4, HexPlayerHuman(), HexPlayerHuman())
    game.play()
