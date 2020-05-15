from enum import Enum, auto


class HexColour(Enum):
    RED = auto()
    BLUE = auto()

    def __neg__(self):
        if self == HexColour.RED:
            return HexColour.BLUE
        if self == HexColour.BLUE:
            return HexColour.RED

    def __str__(self):
        if self == HexColour.RED:
            return 'RED'
        if self == HexColour.BLUE:
            return 'BLUE'

    def __int__(self):
        if self == HexColour.RED:
            return 1
        if self == HexColour.BLUE:
            return -1
