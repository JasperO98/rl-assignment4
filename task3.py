import setup
from hex.tournament import HexTournament
from hex.players.selfplay import AlphaZeroSelfPlay1
from hex.players.base import HexPlayerRandom

if __name__ == '__main__':
    players = [HexPlayerRandom()]

    for hashed in (
            -6518895601560197144,
            7184645914559508968,
            -8068969104488914968,
            -2531536123151864856,
            -3548057543391555352,
            8337567419166355944,
    ):
        players.append(AlphaZeroSelfPlay1(hashed))

    ht = HexTournament(5, players)
    ht.train()
    ht.tournament()
    ht.plots()
