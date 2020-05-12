from hex.board import HexBoard
from alphazero.Game import Game
import numpy as np
from hex.colour import HexColour
from alphazero.NeuralNet import NeuralNet
import os
from keras.models import Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Conv2D, BatchNormalization
from keras.optimizers import Adam


class AlphaHexGame(Game):
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

    def getNextState(self, board, player, action):
        move = divmod(action, self.size)
        assert board[move][0] == 0
        board = np.append(board[:, :, :1], board[:, :, :-1], 2)
        board[move][0] = player
        return board, -player

    def getValidMoves(self, board, player):
        return board[:, :, 0].flatten() == 0

    def getGameEnded(self, board, player):
        tracking = board[:, :, 0] == 1

        if np.all(np.any(tracking, 1)):
            for i in range(1, self.size):
                tracking[i] = np.logical_and(tracking[i], np.logical_or(tracking[i - 1], np.append(tracking[i - 1, 1:], [False])))
                if not np.any(tracking[i]):
                    break
            else:
                return player

        tracking = board[:, :, 0] == -1

        if np.all(np.any(tracking, 0)):
            for i in range(1, self.size):
                tracking[:, i] = np.logical_and(tracking[:, i], np.logical_or(tracking[:, i - 1], np.append(tracking[1:, i - 1], [False])))
                if not np.any(tracking[:, i]):
                    break
            else:
                return -player

        return 0

    def getCanonicalForm(self, board, player):
        return board * player

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
        os.makedirs(name=folder, exist_ok=True)
        self.model.save_weights(os.path.join(folder, filename))

    def load_checkpoint(self, folder, filename):
        self.model.load_weights(os.path.join(folder, filename))

    @staticmethod
    def exists_checkpoint(folder, filename):
        return os.path.exists(os.path.join(folder, filename))
