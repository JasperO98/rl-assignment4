import setup
from hex.tournament import HexTournament
from hex.players.selfplay import AlphaZeroSelfPlay1


def irange(start, stop):
    return range(start, stop + 1)


if __name__ == '__main__':
    players = []
    for i in irange(1, 20):
        players.append(AlphaZeroSelfPlay1())
        players[-1].coach_args.cpuct = i

    ht = HexTournament(5, players)
    ht.train()
    ht.tournament()
    ht.plots()
