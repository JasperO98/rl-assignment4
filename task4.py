from itertools import combinations
from math import log, ceil, floor
from time import time
import matplotlib.pyplot as plt
from trueskill import rate_1vs1
from hex.game import HexGame
from hex.players.montecarlo import HexPlayerMonteCarloIterations
from hex.colour import HexColour
import numpy as np
import csv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def scat_plot(ratings, names, time_list, plot_title=''):
    print(ratings)
    print(names)
    print(time_list)
    plt.clf()
    plt.figure(figsize=(20, 15))
    y = ratings
    x = time_list
    for i in range(len(names)):
        plt.plot(x[i], y[i], linestyle='none', marker='o', label=names[i])

    plt.legend(numpoints=1)

    plt.ylabel('TrueSkill value')
    plt.xlabel('Average game duration in seconds')
    plt.title(plot_title)

    plt.xticks(range(floor(min(x)), ceil(max(x)), 10))

    plt.tight_layout()
    plt.savefig('figures/task4.pdf')

    rows = [ratings, [n[5:] for n in names], time_list]
    with open('figures/test.csv', "w", encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def match(players, p1, p2, n_games, size):
    for _ in range(n_games):
        winner = HexGame(size, players[p1], players[p2]).play([])
        if winner == HexColour.RED:
            yield p1, p2
        elif winner == HexColour.BLUE:
            yield p2, p1


def gameplay(data):
    players, p1, p2, n_runs, board_size = data
    arr = []

    start = time()
    for i in match(players, p1, p2, n_runs, board_size):
        arr.append(i)
    done = time()
    time_taken1 = done - start

    start = time()
    for j in match(players, p2, p1, n_runs, board_size):
        arr.append(j)
    done = time()
    time_taken2 = done - start

    return arr, time_taken1 + time_taken2


def main():
    board_size = 4
    N = np.round(np.geomspace(1, 512, 10)).astype(int)  # range(100, 600, 250)
    Cp = np.linspace(0, 2, 7)

    players = [HexPlayerMonteCarloIterations(n, c) for n in N for c in Cp]
    print(N)
    print(Cp)
    print('No players:', len(players))
    print('No games:', len(list(combinations(players, 2))))
    n_runs = ceil(2 * log(len(players), 2))
    time_list = [0] * len(players)
    comb = list(combinations(range(len(players)), 2))
    with Pool(cpu_count() - 1) as pool:
        for arr, t in tqdm(
                iterable=pool.imap(gameplay, [(players, c[0], c[1], n_runs, board_size) for c in comb]),
                total=len(comb),
        ):
            for a in arr:
                players[a[0]].rating, players[a[1]].rating = \
                    rate_1vs1(players[a[0]].rating, players[a[1]].rating, drawn=False)
            time_list[arr[0][0]] += t / (n_runs * 2) / (len(players) - 1)
            time_list[arr[0][1]] += t / (n_runs * 2) / (len(players) - 1)

    names = [player.__str__() for player in players]
    ratings = [player.rating.mu - 3 * player.rating.sigma for player in players]

    scat_plot(ratings, names, time_list)


if __name__ == '__main__':
    main()
