from hex.players.montecarlo import HexPlayerMonteCarloIterations
import numpy as np
from hex.tournament import HexTournament
from itertools import combinations


def main():
    N = np.round(np.geomspace(1, 512, 10)).astype(int)
    Cp = np.around(np.linspace(0, 2, 7), 2)
    players = tuple(HexPlayerMonteCarloIterations(n, c) for n in N for c in Cp)

    print('Simulations:', N)
    print('Cp:', Cp)
    print()
    print('Number of players:', len(players))
    print('Simulating games:')

    ht = HexTournament(4, players)
    ht.tournament()
    ht.task4()


if __name__ == '__main__':
    main()
