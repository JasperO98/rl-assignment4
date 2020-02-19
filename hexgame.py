from hexboard import HexBoard
from hexplayer import HexPlayer


class HexGame:
    def __init__(self, size):
        self.board = HexBoard(size)
        self.player1 = HexPlayer()
        self.player2 = HexPlayer()

    def play(self):
        while not self.board.is_game_over():
            self.board.render(1000)
            move = self.player2.get_move(self.board) if self.board.moves % 2 else self.player1.get_move(self.board)
            self.board.do_move(move)
