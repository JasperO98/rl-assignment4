from hex.players.base import HexPlayer
from hex.alphazero import AlphaHexGame, AlphaHexNN
from alphazero.Coach import Coach
import numpy as np
import shutil
from os.path import join, split, splitext
from natsort import natsorted
from glob import glob


class CoachArgs:
    def __init__(self, name):
        self.numIters = 1000
        self.maxlenOfQueue = 200000
        self.numEps = 100
        self.tempThreshold = 15
        self.numMCTSSims = 25
        self.cpuct = 1
        self.numItersForTrainExamplesHistory = 20
        self.arenaCompare = 40
        self.updateThreshold = 0.6

        self.checkpoint = 'models/' + name
        self.load_folder_file = (
            self.checkpoint,
            splitext(split(natsorted(glob(join(self.checkpoint, '*.examples')))[-1])[-1])[0],
        )


class AlphaZeroSelfPlay1(HexPlayer):
    NAME = 'player1'

    def __init__(self):
        super().__init__()
        self.args = CoachArgs(self.NAME)
        self.net = None

    def setup(self, size):
        self.net = AlphaHexNN(size)

        if self.net.exists_checkpoint(self.args.checkpoint, 'best.pth.tar'):
            self.net.load_checkpoint(self.args.checkpoint, 'best.pth.tar')
            return

        coach = Coach(AlphaHexGame(size), self.net, self.args)

        if self.net.exists_checkpoint(self.args.checkpoint, 'temp.pth.tar'):
            self.net.load_checkpoint(self.args.checkpoint, 'temp.pth.tar')
            coach.loadTrainExamples()
        else:
            shutil.rmtree(path=self.args.checkpoint, ignore_errors=True)

        coach.learn()

    def determine_move(self, board, renders):
        if self.net is None:
            self.setup(board.size)

        np_board = np.zeros((board.size, board.size))
        for key, value in board.board.items():
            if value == board.turn():
                np_board[key] = 1
            else:
                np_board[key] = -1

        action = int(np.argmax(self.net.predict(np_board)[0] * (np_board.flatten() == 0)))
        return divmod(action, board.size)

    def __str__(self):
        pass


class AlphaZeroSelfPlay2(AlphaZeroSelfPlay1):
    NAME = 'player2'
