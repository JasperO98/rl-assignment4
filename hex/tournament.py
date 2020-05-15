from math import log, ceil
from hex.game import HexGame
from hex.colour import HexColour
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import permutations
from trueskill import Rating, rate_1vs1
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.cm import ScalarMappable
from os.path import join

# disable GPU usage during tournaments
tf.config.set_visible_devices([], 'GPU')


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
            return pi1, pi2, np.mean(winner.active), np.mean(loser.active)
        if winner.colour == HexColour.BLUE:
            return pi2, pi1, np.mean(winner.active), np.mean(loser.active)

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

    @staticmethod
    def _save_plot(name):
        plt.tight_layout()
        plt.savefig(join('figures', name + '.pdf'))
        plt.close()

    def plots(self):
        plt.figure(figsize=(16, 8))
        mapper = ScalarMappable(cmap='Greys')
        plt.bar(
            x=[str(player) for player in self.players],
            height=[ratings[-1].mu for ratings in self.ratings],
            yerr=[ratings[-1].sigma for ratings in self.ratings],
            color=mapper.to_rgba(self.durations),
            edgecolor='black',
            ecolor='red',
            capsize=20,
        )
        plt.colorbar(mapper).set_label('seconds per turn')
        self._save_plot('tournament_ratings')

        plt.figure(figsize=(16, 8))
        for player, ratings in zip(self.players, self.ratings):
            x = range(len(ratings))
            plt.plot(
                x, [rating.mu for rating in ratings],
                label=str(player),
            )
            plt.fill_between(
                x=x,
                y1=[rating.mu - rating.sigma for rating in ratings],
                y2=[rating.mu + rating.sigma for rating in ratings],
                alpha=0.5,
            )
        plt.legend()
        self._save_plot('tournament_convergence')
