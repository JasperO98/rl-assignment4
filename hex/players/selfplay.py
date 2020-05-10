from hex.players.base import HexPlayer
from hex.alphazero import AlphaHexGame, AlphaHexNN
from alphazero.Coach import Coach
import numpy as np
import shutil
from alphazero.MCTS import MCTS
import numpy.random as npr


class ArgsCoach:
    def __init__(self):
        self.numIters = 100
        self.maxlenOfQueue = 200000
        self.numEps = 100
        self.tempThreshold = 15
        self.numMCTSSims = 50
        self.cpuct = 2
        self.numItersForTrainExamplesHistory = 20
        self.arenaCompare = 40
        self.updateThreshold = 0.5
        self.batch_size = 64
        self.epochs = 10
        self.checkpoint = None

    def init(self, size, name):
        self.checkpoint = 'models/' + str(size) + 'x' + str(size) + '/' + str(hash(self)) + '/' + name

    def __hash__(self):
        return hash((
            self.numIters,
            self.maxlenOfQueue,
            self.numEps,
            self.tempThreshold,
            self.numMCTSSims,
            self.cpuct,
            self.numItersForTrainExamplesHistory,
            self.arenaCompare,
            self.updateThreshold,
            self.batch_size,
            self.epochs,
        ))


class ArgsMCTS:
    def __init__(self):
        self.numMCTSSims = 100
        self.cpuct = 1


class AlphaZeroSelfPlay1(HexPlayer):
    NAME = 'player1'

    def __init__(self):
        super().__init__()
        self.coach_args = ArgsCoach()
        self.mcts_args = ArgsMCTS()
        self.mcts_class = None

    def setup(self, size):
        self.coach_args.init(size, self.NAME)
        game = AlphaHexGame(size)
        net = AlphaHexNN(game, self.coach_args)

        if not net.exists_checkpoint(self.coach_args.checkpoint, 'best.pth.tar'):
            shutil.rmtree(self.coach_args.checkpoint, True)
            coach = Coach(game, net, self.coach_args)
            coach.learn()

        net.load_checkpoint(self.coach_args.checkpoint, 'best.pth.tar')
        self.mcts_class = MCTS(game, net, self.mcts_args)

    def determine_move(self, board, renders):
        if self.mcts_class is None:
            self.setup(board.size)

        np_board = np.zeros((board.size, board.size))
        for key, value in board.board.items():
            if value == board.turn():
                np_board[key] = 1
            else:
                np_board[key] = -1

        pi = self.mcts_class.getActionProb(np_board, temp=0)
        action = npr.choice(len(pi), p=pi)
        return divmod(action, board.size)

    def __str__(self):
        pass


class AlphaZeroSelfPlay2(AlphaZeroSelfPlay1):
    NAME = 'player2'
