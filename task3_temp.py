from itertools import combinations
from math import log, ceil
from time import time
import matplotlib.pyplot as plt
from trueskill import rate_1vs1
from hex.game import HexGame
from hex.players.alphabeta import HexPlayerEnhancedAB
from hex.players.montecarlo import HexPlayerMonteCarloTime
from hex.colour import HexColour
from hex.players.base import HexPlayerRandom
import numpy as np
import matplotlib.patches as mpatches
from hex.players.montecarlo import HexPlayerMonteCarloTime, HexPlayerMonteCarloIterations

def bar_plot(ratings, names, plot_title=''):
    y = ratings
    x = names
    plt.bar(x, y)
    plt.ylabel('TrueSkill value')
    plt.title(plot_title)
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(y):
        plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
    plt.xticks(xlocs, x)
    plt.show()
    #plt.savefig('figures/ratings.pdf')

def match(player1, player2, n_games, size):
    LOG_player1 = []
    LOG_player2 = []
    for _ in range(n_games):
        winner = HexGame(size, player1, player2).play([])
        print(player1)
        if winner == HexColour.RED:
            player1.rating, player2.rating = rate_1vs1(player1.rating, player2.rating, drawn=False)
        elif winner == HexColour.BLUE:
            player2.rating, player1.rating = rate_1vs1(player2.rating, player1.rating, drawn=False)
        print(player2)
        LOG_player1.append(player1.mu - 3 * player1.sigma)
        LOG_player2.append(player2.mu - 3 * player2.sigma)
    return player1.rating, player2.rating, [LOG_player1, LOG_player2]

def line_plot(dict):
    legend = []
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]
    for a,b in dict.items():
        x = np.arange(0, len(b))
        y = b
        color_line = colors.pop()
        plt.plot(x, y, color=color_line, linewidth=2)
        color_patch = mpatches.Patch(color = color_line, label = a)
        legend.append(color_patch)
    plt.legend(handles = legend)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Title")
    plt.show()

def main():
    board_size = 2
    players = [
#        HexPlayerEnhancedAB(10, True),
#        HexPlayerEnhancedAB(10, False),
        HexPlayerMonteCarloTime(10, 1),
        HexPlayerRandom()]
    print(players[1], "test")
    n_runs = ceil(2 * log(len(players), 2))
    n_runs = 1
    start = time()
    ELO_LOG = {player: [] for player in [player.__str__() for player in players]}
    for player1, player2 in combinations(players, 2):
        print(player1.__str__(), player2.__str__())
        print(player1.mu)
        player1.rating, player2.rating, log_player1 = match(player1, player2, n_runs, board_size)
        player2.rating, player1.rating, log_player2 = match(player2, player1, n_runs, board_size)
    done = time()
    time_taken = done - start
    names = [player.__str__() for player in players]
    ratings = [player.rating.mu - 3 * player.rating.sigma for player in players]
    print('time in seconds\t\t:', time_taken)

    bar_plot(ratings, names)


    ELO_LOG2 = {"A" : log_player1[0] + log_player2[1],
                "B" : log_player1[1] + log_player2[0]}
    line_plot(ELO_LOG2)

if __name__ == '__main__':
    main()
