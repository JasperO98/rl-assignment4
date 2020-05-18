import setup
from hex.tournament import HexTournament
from hex.players.selfplay import AlphaZeroSelfPlay1


def irange(start, stop, step=1):
    return range(start, stop + 1, step)


if __name__ == '__main__':
    players = []
    for i in irange(1, 20):
        players.append(AlphaZeroSelfPlay1())
        players[-1].coach_args.epochs = i
    print(players)

    ht = HexTournament(5, players)
    ht.train()
    ht.tournament()
    ht.plots()
