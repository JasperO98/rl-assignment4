from hex.players.base import HexPlayer
from hex.alphazero import AlphaHexGame, AlphaHexNN
from alphazero.Coach import Coach
import numpy as np
import shutil
from alphazero.MCTS import MCTS


class CoachArgs:
    def __init__(self):
        self.numIters = 100
        self.maxlenOfQueue = 200000
        self.numEps = 100
        self.tempThreshold = 15
        self.numMCTSSims = 50
        self.cpuct = 1
        self.numItersForTrainExamplesHistory = 20
        self.arenaCompare = 40
        self.updateThreshold = 0.5
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
        ))


class AlphaZeroSelfPlay1(HexPlayer):
    NAME = 'player1'

    def __init__(self):
        super().__init__()
        self.args = CoachArgs()
        self.net = None
        self.game = None
        self.mcts = None

    def setup(self, size):
        self.args.init(size, self.NAME)
        self.game = AlphaHexGame(size)
        self.net = AlphaHexNN(self.game)
        self.mcts = MCTS(self.game, self.net, self.args)  # For playing these args might change, not CoachArgs()

        if not self.net.exists_checkpoint(self.args.checkpoint, 'best.pth.tar'):
            shutil.rmtree(self.args.checkpoint, True)
            coach = Coach(self.game, self.net, self.args)
            coach.learn()

        self.net.load_checkpoint(self.args.checkpoint, 'best.pth.tar')

    def determine_move(self, board, renders):
        if self.net is None:
            self.setup(board.size)

        np_board = np.zeros((board.size, board.size))
        for key, value in board.board.items():
            if value == board.turn():
                np_board[key] = 1
            else:
                np_board[key] = -1
        
        pi = self.mcts.getActionProb(np_board, temp=0)  # Activates MCTS

        action = np.random.choice(len(pi), p=pi)
        print(action)

        # action = int(np.argmax(self.net.predict(np_board)[0] * (np_board.flatten() == 0)))
        return divmod(action, board.size)

    def __str__(self):
        pass


class AlphaZeroSelfPlay2(AlphaZeroSelfPlay1):
    NAME = 'player2'
