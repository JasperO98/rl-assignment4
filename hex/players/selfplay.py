from hex.players.base import HexPlayer
from alphazero.Coach import Coach
import numpy as np
import shutil
from alphazero.MCTS import MCTS
import numpy.random as npr
from copy import deepcopy
from alphazero.Game import Game
from alphazero.NeuralNet import NeuralNet
import os
from keras.models import Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Conv2D, BatchNormalization
from keras.optimizers import Adam
from scipy.ndimage.measurements import label
import json
from subprocess import run
from filelock import FileLock
from keras.utils import plot_model


class AlphaHexGame(Game):
    CONNECTIVITY = ((0, 1, 1), (1, 1, 1), (1, 1, 0))

    def __init__(self, size, args):
        super().__init__()
        self.size = size
        self.args = args

    def getInitBoard(self):
        return np.zeros((self.size, self.size, self.args.depth), int)

    def getBoardSize(self):
        return self.size, self.size, self.args.depth

    def getActionSize(self):
        return self.size * self.size

    def actionToCoordinates(self, player, action):
        move = divmod(action, self.size)
        if player == -1:
            move = move[::-1]
        return move

    def getNextState(self, board, player, action):
        board = np.append(board[:, :, :1], board[:, :, :-1], 2)
        board[self.actionToCoordinates(player, action)][0] = player
        return board, -player

    def getValidMoves(self, board, player):
        return board[:, :, 0].flatten() == 0

    def getGameEnded(self, board, player):
        labeled = board[:, :, 0] == 1
        if np.all(np.any(labeled, 1)):
            labeled = label(labeled, self.CONNECTIVITY)[0]
            if len(set(labeled[0]).difference([0]).intersection(labeled[-1])) > 0:
                return player

        labeled = board[:, :, 0] == -1
        if np.all(np.any(labeled, 0)):
            labeled = label(labeled, self.CONNECTIVITY)[0]
            if len(set(labeled[:, 0]).difference([0]).intersection(labeled[:, -1])) > 0:
                return -player

        return 0

    def getCanonicalForm(self, board, player):
        if player == 1:
            return board
        return np.flip(np.rot90(-board), 0)

    def getSymmetries(self, board, pi):
        return (
            (board, pi),
            (np.rot90(board, 2), np.flip(pi)),
        )

    def stringRepresentation(self, board):
        return board[:, :, 0].tostring()


class AlphaHexNN(NeuralNet):
    def build_model(self):
        inputs = Input(self.input)

        layer = Activation('relu')(BatchNormalization()(Conv2D(filters=512, kernel_size=3, padding='same')(inputs)))
        for _ in range((min(self.input[:2]) - 1) // 2):
            layer = Activation('relu')(BatchNormalization()(Conv2D(filters=512, kernel_size=3, padding='valid')(layer)))
        layer = Flatten()(layer)
        layer = Dropout(0.3)(Activation('relu')(BatchNormalization()(Dense(1024)(layer))))
        layer = Dropout(0.3)(Activation('relu')(BatchNormalization()(Dense(1024)(layer))))

        pi = Dense(units=self.output, activation='softmax', name='pi')(layer)
        v = Dense(units=1, activation='tanh', name='v')(layer)

        model = Model(inputs=inputs, outputs=(pi, v))
        model.compile(optimizer=Adam(), loss=('categorical_crossentropy', 'mean_squared_error'))

        plot_model(model, os.path.join('figures', 'model.' + str(self.input[0]) + 'x' + str(self.input[1]) + '.pdf'))
        return model

    def __init__(self, game):
        super().__init__(game)
        self.args = game.args
        self.input = game.getBoardSize()
        self.output = game.getActionSize()
        self.model = self.build_model()

    def train(self, examples):
        boards, pis, vs = list(zip(*examples))
        self.model.fit(
            x=np.array(boards),
            y=[np.array(pis), np.array(vs)],
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
        )

    def predict(self, board):
        pis, vs = self.model.predict(np.expand_dims(board, 0))
        return pis[0], vs[0]

    def save_checkpoint(self, folder, filename):
        # ensure data folder exists
        os.makedirs(name=folder, exist_ok=True)
        # save model checkpoints
        self.model.save_weights(os.path.join(folder, filename))
        # save model parameters
        with open(os.path.join(folder, 'parameters.json'), 'w') as fp:
            json.dump(obj=self.args.json(), fp=fp, indent=2)

    def load_checkpoint(self, folder, filename):
        self.model.load_weights(os.path.join(folder, filename))

    @staticmethod
    def exists_checkpoint(folder, filename):
        return os.path.exists(os.path.join(folder, filename))


class ArgsCoach:
    def __init__(self, hashed=False):
        self.hashed = hashed
        # iteration parameters
        self.numIters = 50
        self.numEps = 50
        # hyperparameters for MCTS
        self.numMCTSSims = 100
        self.cpuct = 5
        # parameters for accepting new networks
        self.arenaCompare = 10
        self.updateThreshold = 0.51
        # parameters for NN training
        self.maxlenOfQueue = None
        self.numItersForTrainExamplesHistory = max(1, self.numIters // 10)
        self.batch_size = 64
        self.epochs = 10
        # board shape parameters
        self.depth = 3
        self.tempThreshold = None
        # disk io parameters
        self.checkpoint = None

    def json(self):
        data = deepcopy(self.__dict__)
        del data['checkpoint']
        del data['hashed']
        del data['maxlenOfQueue']
        return data

    def init(self, size, name):
        self.tempThreshold = size * 2 - 1
        self.checkpoint = 'models/' + str(size) + 'x' + str(size) + '/' + str(hash(self)) + '/' + name

    def __hash__(self):
        if self.hashed:
            return self.hashed

        return hash((
            self.numIters,
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
        self.numMCTSSims = 1000
        self.cpuct = 1


class AlphaZeroSelfPlay1(HexPlayer):
    NAME = 'player1'

    def __init__(self, hashed=False):
        super().__init__()
        self.coach_args = ArgsCoach(hashed)
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

            with FileLock('.gitlocked'):
                run(('git', 'add', os.path.join(self.coach_args.checkpoint, 'best.pth.tar')))
                run(('git', 'add', os.path.join(self.coach_args.checkpoint, 'parameters.json')))
                run((
                    'git', 'commit', '-m',
                    'add trained model (' + str(size) + 'x' + str(size) + ', ' + self.NAME + ', ' + str(hash(self.coach_args)) + ')',
                ))
                run(('git', 'pull', '--no-edit'))
                run(('git', 'push'))

        net.load_checkpoint(self.coach_args.checkpoint, 'best.pth.tar')
        self.mcts_class = MCTS(game, net, self.mcts_args)

    def determine_move(self, board, renders):
        if self.mcts_class is None:
            self.setup(board.size)

        np_board = np.zeros((board.size, board.size, self.coach_args.depth), int)
        for i, ((x, y), colour) in enumerate(board.board.items()):
            z = range(min(self.coach_args.depth, len(board.board) - i))
            np_board[x, y, z] = int(colour)
        np_board = self.mcts_class.game.getCanonicalForm(np_board, int(self.colour))

        pi = self.mcts_class.getActionProb(np_board, 0)
        action = npr.choice(a=len(pi), p=pi)
        return self.mcts_class.game.actionToCoordinates(int(self.colour), action)

    def __str__(self):
        return 'AlphaZero Player ' + self.NAME[-1] + '\n' + str(hash(self.coach_args))


class AlphaZeroSelfPlay2(AlphaZeroSelfPlay1):
    NAME = 'player2'
