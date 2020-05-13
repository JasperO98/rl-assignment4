from hex.players.base import HexPlayer
from alphazero.Coach import Coach
import numpy as np
import shutil
from alphazero.MCTS import MCTS
import numpy.random as npr
from hex.colour import HexColour
from copy import deepcopy
from alphazero.Game import Game
from alphazero.NeuralNet import NeuralNet
import os
from keras.models import Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Conv2D, BatchNormalization
from keras.optimizers import Adam
from scipy.ndimage.measurements import label
import json


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
        move = self.actionToCoordinates(player, action)
        assert board[move][0] == 0
        board = np.append(board[:, :, :1], board[:, :, :-1], 2)
        board[move][0] = player
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

        return model

    def __init__(self, game):
        super().__init__(game)
        self.args = game.args
        self.input = game.getBoardSize()
        self.output = game.getActionSize()
        self.model = self.build_model()

    def train(self, examples):
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=self.args.batch_size, epochs=self.args.epochs)

    def predict(self, board):
        board = board[np.newaxis, :, :]
        pi, v = self.model.predict(board)
        return pi[0], v[0]

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
    def __init__(self, epochs, cp, episodes, threshold):
        self.numIters = 100
        self.maxlenOfQueue = 200000
        self.numEps = episodes
        self.tempThreshold = 15
        self.numMCTSSims = 50
        self.cpuct = cp
        self.numItersForTrainExamplesHistory = 20
        self.arenaCompare = 40
        self.updateThreshold = threshold
        self.batch_size = 64
        self.epochs = epochs
        self.depth = 3
        self.checkpoint = None

    def json(self):
        data = deepcopy(self.__dict__)
        del data['checkpoint']
        return data

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

    def __init__(self, epochs, cp, episodes, threshold):
        super().__init__()
        self.coach_args = ArgsCoach(epochs, cp, episodes, threshold)
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
        return 'AlphaZero Player ' + self.NAME[-1] + '\n' + str(hash(self.coach_args))


class AlphaZeroSelfPlay2(AlphaZeroSelfPlay1):
    NAME = 'player2'
