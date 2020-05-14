import re
from abc import ABC, abstractmethod
from time import time
from random import choice


class HexPlayer(ABC):
    def __init__(self):
        self.active = 0
        self.colour = None

    def __repr__(self):
        return str(self).replace('\n', ' ')

    def get_move(self, board, renders):
        start = time()
        move = self.determine_move(board, renders)
        self.active += time() - start
        return move

    @abstractmethod
    def determine_move(self, board, renders):
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
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def determine_move(self, board, renders):
        if self.name is not None:
            print('-> ' + self.name + '\'s turn!')

        while True:
            move = self.string_to_move(input('Coordinates: ').lower())
            if not board.exists(move) or not board.is_empty(move):
                print('Invalid Move')
                continue
            return move

    def __str__(self):
        return 'Human Player' + ('' if self.name is None else '\n\'' + self.name + '\'')


class HexPlayerRandom(HexPlayer):
    def determine_move(self, board, renders):
        return choice(list(board.moves()))

    def __str__(self):
        return 'Random Player'
