import re
from abc import ABC, abstractmethod


class HexPlayer(ABC):
    @abstractmethod
    def get_move(self, board):
        pass


class HexPlayerHuman(HexPlayer):
    def get_move(self, board):
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
