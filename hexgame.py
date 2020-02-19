from hexboard import HexBoard
from hexcolour import HexColour


class HexGame:
    def __init__(self, size, player1, player2):
        self.board = HexBoard(size)
        self.player1 = player1
        self.player2 = player2

    def play(self):
        while not self.board.is_game_over():
            self.board.render(1000)
            move = self.player2.get_move(self.board) if self.board.moves % 2 else self.player1.get_move(self.board)
            self.board.do_move(move)

        self.board.render(1000)
        if self.board.check_win(HexColour.RED):
            print('Red Wins!')
        if self.board.check_win(HexColour.BLUE):
            print('Blue Wins!')
