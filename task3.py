import setup
from hex.tournament import HexTournament
from hex.players.selfplay import AlphaZeroSelfPlay1

if __name__ == '__main__':
    ht = HexTournament(5, [AlphaZeroSelfPlay1(cp=cp) for cp in range(1, 21)])
    ht.tournament()
    ht.plots()
