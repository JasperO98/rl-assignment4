from math import log, ceil
from hex.game import HexGame
from hex.colour import HexColour
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import permutations
from trueskill import Rating, rate_1vs1
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class HexTournament:
    def __init__(self, size, players):
        self.size = size
        self.players = players
        self.durations = np.zeros(len(players), float)
        self.ratings = [[Rating()] for _ in range(len(players))]

    def _match_unpack(self, args):
        return self.match(*args)

    def match(self, pi1, pi2):
        winner, loser = HexGame(self.size, self.players[pi1], self.players[pi2]).play([])
        if winner.colour == HexColour.RED:
            return pi1, pi2, winner.active, loser.active
        if winner.colour == HexColour.BLUE:
            return pi2, pi1, winner.active, loser.active

    def tournament(self):
        matches = list(permutations(range(len(self.players)), 2)) * ceil(2 * log(len(self.players), 2))
        with Pool(cpu_count() - 1) as pool:
            for wi, li, wd, ld in tqdm(iterable=pool.imap(self._match_unpack, matches), total=len(matches)):
                # update ratings
                wr, lr = rate_1vs1(self.ratings[wi][-1], self.ratings[li][-1])
                self.ratings[wi].append(wr)
                self.ratings[li].append(lr)
                # update durations
                self.durations[wi] += wd / sum(wi in match for match in matches)
                self.durations[li] += ld / sum(li in match for match in matches)

    def plot_elo(self):
        sns.barplot(
            x=[str(player) for player in self.players],
            y=[rating[-1].mu for rating in self.ratings],
            ci=[rating[-1].sigma for rating in self.ratings],
        )
        plt.show()
