from itertools import combinations
from math import log, ceil
from time import time
import matplotlib.pyplot as plt
from trueskill import rate_1vs1
from hexgame import HexGame
from hexplayer import HexPlayerDijkstra, HexPlayerRandom, HexPlayerEnhanced


def bar_plot(ratings, names, plot_title=""):
    y = ratings
    x = names
    plt.bar(x, y)
    plt.ylabel('TrueSkill value')
    plt.title(plot_title)
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(y):
        plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
    plt.xticks(xlocs, x)
    plt.savefig('ratings.pdf')


def match(player1, player2, n_games, size):
    for _ in range(n_games):
        game = HexGame(size, player1, player2)
        game.play([''])
        if game.win[1] == 1:
            player1.rating, player2.rating = rate_1vs1(player1.rating, player2.rating, drawn=False)
        elif game.win[1] == 2:
            player2.rating, player1.rating = rate_1vs1(player2.rating, player1.rating, drawn=False)
    return player1.rating, player2.rating


def main():
    board_size = 4
    players = [HexPlayerDijkstra(3),
               HexPlayerDijkstra(4),
               HexPlayerRandom(3),
               HexPlayerEnhanced(10, True),
               HexPlayerEnhanced(10, False)]
    n_runs = ceil(2 * log(len(players), 2))
    start = time()
    for player1, player2 in combinations(players, 2):
        player1.rating, player2.rating = match(player1, player2, n_runs, board_size)
        player2.rating, player1.rating = match(player2, player1, n_runs, board_size)
    done = time()
    time_taken = done - start
    names = [player.__str__() for player in players]
    ratings = [player.rating.mu - 3 * player.rating.sigma for player in players]
    print("time in seconds\t\t:", time_taken)
    bar_plot(ratings, names)


if __name__ == "__main__":
    main()
