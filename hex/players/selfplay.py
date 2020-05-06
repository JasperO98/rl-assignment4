from hex.players.base import HexPlayer
from hex.alphazero import AlphaHexGame, AlphaHexNN
from alphazero.Coach import Coach
from alphazero.utils import dotdict


class AlphaZeroSelfPlay(HexPlayer):
    def __init__(self):
        super().__init__()

        game = AlphaHexGame(7)
        nnet = AlphaHexNN(game)

        coach = Coach(game, nnet, dotdict({
            'numIters': 1000,
            'numEps': 100,
            'tempThreshold': 15,
            'updateThreshold': 0.6,
            'maxlenOfQueue': 200000,
            'numMCTSSims': 25,
            'arenaCompare': 40,
            'cpuct': 1,
            'checkpoint': 'models/player1',
            'load_model': False,
            'numItersForTrainExamplesHistory': 20,
        }))
        coach.learn()

    def determine_move(self, board, renders):
        pass

    def __str__(self):
        pass
