import setup
from hex.players.selfplay import AlphaZeroSelfPlay
import numpy as np
from hex.board import HexBoard


if __name__ == '__main__':
    AlphaZeroSelfPlay()
    # AlphaZeroSelfPlay.determine_move(board)