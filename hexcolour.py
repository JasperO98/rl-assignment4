from enum import Enum, auto


class HexColour(Enum):
    RED = auto()
    BLUE = auto()

    def invert(self):
        if self == HexColour.RED:
            return HexColour.BLUE
        else:
            return HexColour.RED
