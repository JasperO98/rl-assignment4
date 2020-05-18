import setup
from hex.tournament import HexTournament
from hex.players.selfplay import AlphaZeroSelfPlay1


def irange(start, stop, step=1):
    return range(start, stop + 1, step)


if __name__ == '__main__':
    players = []
    for i in (0, 0.5, 0.51, 0.55, 0.6, 0.7):
        players.append(AlphaZeroSelfPlay1())
        players[-1].coach_args.updateThreshold = i
    print(players)

    ht = HexTournament(5, players)
    ht.train()
    ht.tournament()
    ht.plots()
