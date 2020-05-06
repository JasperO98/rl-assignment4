from hex.players.base import HexPlayer
from hex.alphazero import AlphaHexGame, AlphaHexNN
from alphazero.Coach import Coach
from alphazero.utils import dotdict
import numpy as np
import shutil


class AlphaZeroSelfPlay(HexPlayer):
    def __init__(self):
        super().__init__()

        self.args = dotdict({
            'numIters': 1000,
            'maxlenOfQueue': 200000,
            'numEps': 100,
            'tempThreshold': 15,
            'numMCTSSims': 25,
            'cpuct': 1,
            'numItersForTrainExamplesHistory': 20,
            'arenaCompare': 40,
            'updateThreshold': 0.6,
            'checkpoint': 'models/player1',
        })
        self.model = None

    def setup(self, size):
        game = AlphaHexGame(size)
        net = AlphaHexNN(game)
        coach = Coach(game, net, self.args)

        if net.exists_checkpoint(self.args.checkpoint, 'best.pth.tar'):
            net.load_checkpoint(self.args.checkpoint, 'best.pth.tar')
            coach.loadTrainExamples()
        else:
            shutil.rmtree(path=self.args.checkpoint, ignore_errors=True)

        coach.learn()
        self.model = net.model

    def determine_move(self, board, renders):
        if self.model is None:
            self.setup(board.size)

        np_board = np.zeros((board.size, board.size))
        for key, value in board.board.items():
            if value == board.turn():
                np_board[key] = 1
            else:
                np_board[key] = -1

        action = int(np.argmax(self.model.predict(np_board) * (np_board.flatten() == 0)))
        return divmod(action, board.size)

    def __str__(self):
        pass
