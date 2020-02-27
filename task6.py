from hexgame import HexGame
from hexplayer import HexPlayerHuman, HexPlayerEnhanced
from hexboard import HexBoard

if __name__ == '__main__':
    # test functionality
    board = HexBoard(2)
    for child, _ in board.children():
        print(child.board, hash(child))

    game = HexGame(2, HexPlayerEnhanced(10, True), None)
    game.step(['tree'])

    # play a game
    game = HexGame(5, HexPlayerEnhanced(10, True), HexPlayerHuman())
    game.play()
