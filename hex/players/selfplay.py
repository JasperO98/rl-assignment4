from hex.players.base import HexPlayer
from hex.alphazero import AlphaHexGame, AlphaHexNN
from alphazero.Coach import Coach
import numpy as np
import shutil
from alphazero.MCTS import MCTS
import numpy.random as npr
from hex.colour import HexColour


class ArgsCoach:
    def __init__(self, epochs, cp):
        self.numIters = 100
        self.maxlenOfQueue = 200000
        self.numEps = 50
        self.tempThreshold = 15
        self.numMCTSSims = 50
        self.cpuct = cp
        self.numItersForTrainExamplesHistory = 20
        self.arenaCompare = 40
        self.updateThreshold = 0.5
        self.batch_size = 64
        self.epochs = epochs
        self.depth = 3
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
            self.depth,
        ))


class ArgsMCTS:
    def __init__(self):
        self.numMCTSSims = 100
        self.cpuct = 1


class AlphaZeroSelfPlay1(HexPlayer):
    NAME = 'player1'

    def __init__(self, epochs, cp):
        super().__init__()
        self.coach_args = ArgsCoach(epochs, cp)
        self.mcts_args = ArgsMCTS()
        self.mcts_class = None

    def setup(self, size):
        self.coach_args.init(size, self.NAME)
        game = AlphaHexGame(size, self.coach_args)
        net = AlphaHexNN(game)

        if not net.exists_checkpoint(self.coach_args.checkpoint, 'best.pth.tar'):
            shutil.rmtree(self.coach_args.checkpoint, True)
            coach = Coach(game, net, self.coach_args)
            coach.learn()

        net.load_checkpoint(self.coach_args.checkpoint, 'best.pth.tar')
        self.mcts_class = MCTS(game, net, self.mcts_args)

    def determine_move(self, board, renders):
        if self.mcts_class is None:
            self.setup(board.size)
        player = 1 if self.colour == HexColour.RED else -1

        np_board = np.zeros((board.size, board.size, self.coach_args.depth), int)
        for i, (x, y, color) in enumerate(board.history[::-1]):
            z = range(min(self.coach_args.depth, i + 1))
            if color == HexColour.RED:
                np_board[x, y, z] = 1
            else:
                np_board[x, y, z] = -1
        np_board = self.mcts_class.game.getCanonicalForm(np_board, player)

        pi = self.mcts_class.getActionProb(np_board, 0)
        action = npr.choice(a=len(pi), p=pi)
        return self.mcts_class.game.actionToCoordinates(player, action)

    def __str__(self):
        return 'AlphaZero Player ' + self.NAME[-1]


class AlphaZeroSelfPlay2(AlphaZeroSelfPlay1):
    NAME = 'player2'
