from enum import Enum, auto


class HexColour(Enum):
    RED = auto()
    BLUE = auto()

    def invert(self):
        if self == HexColour.RED:
            return HexColour.BLUE
        if self == HexColour.BLUE:
            return HexColour.RED

    def __str__(self):
        if self == HexColour.RED:
            return 'RED'
        if self == HexColour.BLUE:
            return 'BLUE'
