import setup
from hex.tournament import HexTournament
from hex.players.selfplay import AlphaZeroSelfPlay1

if __name__ == '__main__':
    ht = HexTournament(5, [AlphaZeroSelfPlay1(depth=depth) for depth in range(1, 11)])
    ht.train()
    ht.tournament()
    ht.plots()
