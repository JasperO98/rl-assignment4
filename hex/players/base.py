import re
from abc import ABC, abstractmethod
from trueskill import TrueSkill


class HexPlayer(ABC):
    ENV = TrueSkill(mu=25, sigma=8 + 1 / 3, draw_probability=0)

    def __init__(self):
        self.rating = HexPlayer.ENV.create_rating()

    @abstractmethod
    def get_move(self, board, colour, renders):
        pass

    @abstractmethod
    def __str__(self):
        pass


class HexPlayerHuman(HexPlayer):
    def get_move(self, board, colour, renders):
        while True:

            match = re.match(r'^([0-9]+)([a-z])$', input('Coordinates: ').lower())

            if not match:
                print('Invalid Move')
                continue

            row = int(match.groups()[0])
            column = ord(match.groups()[1]) - ord('a')

            if not board.exists((row, column)) or not board.is_empty((row, column)):
                print('Invalid Move')
                continue

            return row, column

    def __str__(self):
        return 'Human Player'
