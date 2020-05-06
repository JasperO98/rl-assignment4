from hex.board import HexBoard
from alphazero.Game import Game
import numpy as np
from hex.colour import HexColour
from alphazero.NeuralNet import NeuralNet
from alphazero.utils import dotdict
import os
from keras.models import Model
from keras.layers import Input, Reshape, Activation, Dropout, Flatten, Dense, Conv2D, BatchNormalization
from keras.optimizers import Adam
from copy import deepcopy


class AlphaHexGame(Game):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def getInitBoard(self):
        return np.zeros((self.size, self.size), int)

    def getBoardSize(self):
        return self.size, self.size

    def getActionSize(self):
        return self.size * self.size

    def getNextState(self, board, player, action):
        board = deepcopy(board)
        board[divmod(action, self.size)] = player
        return board, -player

    def getValidMoves(self, board, player):
        return board.flatten() == 0

    def getGameEnded(self, board, player):
        hex_board = HexBoard(self.size)

        hex_board.set_colour(np.argwhere(board == player), HexColour.RED)
        hex_board.set_colour(np.argwhere(board == -player), HexColour.BLUE)

        if hex_board.check_win(HexColour.RED):
            return player
        if hex_board.check_win(HexColour.BLUE):
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
        return board.tostring()


class AlphaHexNN(NeuralNet):
    @staticmethod
    def build_model(x, y, n, args):
        input_boards = Input(shape=(x, y))
        x_image = Reshape((x, y, 1))(input_boards)
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv3)))
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))
        pi = Dense(n, activation='softmax', name='pi')(s_fc2)
        v = Dense(1, activation='tanh', name='v')(s_fc2)
        model = Model(inputs=input_boards, outputs=[pi, v])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
        return model

    def __init__(self, game):
        super().__init__(game)

        self.args = dotdict({
            'lr': 0.001,
            'dropout': 0.3,
            'epochs': 10,
            'batch_size': 64,
            'cuda': False,
            'num_channels': 512,
        })

        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.model = self.build_model(self.board_x, self.board_y, self.action_size, self.args)

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

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ('No model in path {}'.format(filepath))
        self.model.load_weights(filepath)
