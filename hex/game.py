from hex.board import HexBoard
from hex.colour import HexColour
from copy import deepcopy


class HexGame:
    def __init__(self, size, player1, player2):
        assert size % 2 == 1
        self.board = HexBoard(size)
        self.player1 = deepcopy(player1)
        self.player1.colour = HexColour.RED
        self.player2 = deepcopy(player2)
        self.player2.colour = HexColour.BLUE

    def step(self, renders=('board', 'win', 'progress')):
        if 'board' in renders:
            self.board.render(1000)

        for player in (self.player1, self.player2):
            if player.colour == self.board.turn():
                self.board.do_move(player.get_move(self.board, renders))
                break

    def play(self, renders=('board', 'win', 'progress')):
        while not self.board.is_game_over():
            self.step(renders)

        if 'board' in renders:
            self.board.render(1000)

        if self.board.check_win(self.player1.colour):
            if 'win' in renders:
                print(str(self.player1.colour) + ' wins!')
            return self.player1, self.player2

        if self.board.check_win(self.player2.colour):
            if 'win' in renders:
                print(str(self.player2.colour) + ' wins!')
            return self.player2, self.player1
