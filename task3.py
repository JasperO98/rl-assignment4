import setup
from hex.tournament import HexTournament
from hex.players.selfplay import AlphaZeroSelfPlay1

if __name__ == '__main__':
    ht = HexTournament(5, [AlphaZeroSelfPlay1(epochs=epochs) for epochs in range(1, 21)])
    ht.train()
    ht.tournament()
    ht.plots()
