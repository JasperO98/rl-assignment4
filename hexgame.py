from hexboard import HexBoard
from hexcolour import HexColour


class HexGame:
    def __init__(self, size, player1, player2):
        self.board = HexBoard(size)
        self.player1 = player1
        self.player2 = player2
        self.win = ""
        self.lose = ""

    def step(self, renders=('board', 'win')):
        if 'board' in renders:
            self.board.render(1000)

        if self.player1 and self.board.turn() == HexColour.RED:
            move = self.player1.get_move(self.board, HexColour.RED, renders)
        if self.player2 and self.board.turn() == HexColour.BLUE:
            move = self.player2.get_move(self.board, HexColour.BLUE, renders)
        self.board.do_move(move)

    def play(self, renders=('board', 'win')):
        while not self.board.is_game_over():
            self.step(renders)

        if 'board' in renders:
            self.board.render(1000)

        if self.board.check_win(HexColour.RED):
            self.win = [self.player1, 1]
            self.lose = [self.player2, 2]
            if 'win' in renders:
                print('Red Wins!')
            return HexColour.RED

        if self.board.check_win(HexColour.BLUE):
            self.win = [self.player2, 2]
            self.lose = [self.player1, 1]
            if 'win' in renders:
                print('Blue Wins!')
            return HexColour.BLUE
