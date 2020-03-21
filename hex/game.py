from hex.board import HexBoard
from hex.colour import HexColour
from copy import deepcopy


class HexGame:
    def __init__(self, size, player1, player2):
        self.board = HexBoard(size)
        self.player1 = deepcopy(player1)
        self.player1.colour = HexColour.RED
        self.player2 = deepcopy(player2)
        self.player2.colour = HexColour.BLUE

    def step(self, renders=('board', 'win', 'progress')):
        if 'board' in renders:
            self.board.render(1000)

        if self.player1 and self.board.turn() == HexColour.RED:
            move = self.player1.get_move(self.board, HexColour.RED, renders)
        if self.player2 and self.board.turn() == HexColour.BLUE:
            move = self.player2.get_move(self.board, HexColour.BLUE, renders)
        self.board.do_move(move)

    def play(self, renders=('board', 'win', 'progress')):
        while not self.board.is_game_over():
            self.step(renders)

        if 'board' in renders:
            self.board.render(1000)

        if self.board.check_win(HexColour.RED):
            if 'win' in renders:
                print('Red Wins!')
            return HexColour.RED

        if self.board.check_win(HexColour.BLUE):
            if 'win' in renders:
                print('Blue Wins!')
            return HexColour.BLUE
