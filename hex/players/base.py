import re
from abc import ABC, abstractmethod
from trueskill import TrueSkill
from random import choice


class HexPlayer(ABC):
    ENV = TrueSkill(mu=25, sigma=8 + 1 / 3, draw_probability=0)

    def __init__(self):
        self.rating = HexPlayer.ENV.create_rating()

    def __repr__(self):
        return self.__str__().replace('\n', ' ')

    @abstractmethod
    def get_move(self, board, colour, renders):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @staticmethod
    def string_to_move(string):
        match = re.match(r'^([0-9]+)([a-z])$', string)
        if not match:
            return None
        row = int(match.groups()[0])
        column = ord(match.groups()[1]) - ord('a')
        return row, column

    @staticmethod
    def move_to_string(move):
        return str(move[0]) + chr(move[1] + ord('a'))


class HexPlayerHuman(HexPlayer):
    def get_move(self, board, colour, renders):
        while True:
            move = self.string_to_move(input('Coordinates: ').lower())
            if not board.exists(move) or not board.is_empty(move):
                print('Invalid Move')
                continue
            return move

    def __str__(self):
        return 'Human Player'


class HexPlayerRandom(HexPlayer):
    def get_move(self, board, colour, renders):
        return choice(list(board.moves()))

    def __str__(self):
        return 'Random Player'
