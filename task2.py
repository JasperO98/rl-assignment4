from hex.game import HexGame
from hex.player.base import HexPlayerHuman
from hex.player.alphabeta import HexPlayerEnhanced

if __name__ == '__main__':
    # play a game
    game = HexGame(5, HexPlayerEnhanced(9, True), HexPlayerHuman())
    game.play()
