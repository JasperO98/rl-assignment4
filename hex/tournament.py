from hex.game import HexGame
from hex.colour import HexColour
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import permutations, product
from trueskill import Rating, rate_1vs1
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.cm import ScalarMappable
from os.path import join
from hex.players.base import HexPlayerHuman
import cv2 as cv
import numpy.random as npr
import pickle

# disable GPU usage during tournaments
tf.config.set_visible_devices([], 'GPU')


class HexTournament:
    def __init__(self, size, players):
        self.size = size
        self.players = players
        self.durations = [[] for _ in range(len(players))]
        self.ratings = [[Rating()] for _ in range(len(players))]

        self.humans = []
        self.computers = []
        for i in range(len(players)):
            if isinstance(players[i], HexPlayerHuman):
                self.humans.append(i)
            else:
                self.computers.append(i)

    def _match_unpack(self, args):
        return self.match(*args)

    def match(self, pi1, pi2, human):
        game = HexGame(self.size, self.players[pi1], self.players[pi2])

        if human:
            winner, loser = game.play()
        else:
            winner, loser = game.play([])

        if winner.colour == HexColour.RED:
            return pi1, pi2, winner.active, loser.active
        if winner.colour == HexColour.BLUE:
            return pi2, pi1, winner.active, loser.active

    def _matches_human(self):
        for _ in range(2):
            for match in permutations(self.humans, 2):
                yield match
            for match in product(self.humans, self.computers):
                yield match
            for match in product(self.computers, self.humans):
                yield match

    def _matches_computer(self):
        for _ in range(8):
            for match in permutations(self.computers, 2):
                yield match

    def _update_after_match(self, wi, li, wd, ld):
        # update ratings
        wr, lr = rate_1vs1(self.ratings[wi][-1], self.ratings[li][-1])
        self.ratings[wi].append(wr)
        self.ratings[li].append(lr)
        # update durations
        self.durations[wi] += wd
        self.durations[li] += ld

    def tournament(self):
        # matches involving at least one human
        matches = npr.permutation(list(self._matches_human()))
        for i, (pi1, pi2) in enumerate(matches, 1):
            print('Game ' + str(i) + ' out of ' + str(len(matches)) + '.')
            wi, li, wd, ld = self.match(pi1, pi2, True)
            self._update_after_match(wi, li, wd, ld)
        cv.destroyAllWindows()

        # matches between computers only
        matches = np.insert(npr.permutation(list(self._matches_computer())), 2, False, 1)
        with self._make_pool(matches) as pool:
            for wi, li, wd, ld in tqdm(iterable=pool.imap(self._match_unpack, matches), total=len(matches)):
                self._update_after_match(wi, li, wd, ld)

        # save raw tournament results
        with open(join('figures', 'raw.pickle'), 'wb') as fp:
            pickle.dump(self, fp)

    def _train(self, pi):
        try:
            self.players[pi].setup(self.size)
        except (AttributeError, TypeError):
            pass

    def train(self):
        with self._make_pool(self.computers) as pool:
            pool.map(self._train, self.computers)

    @staticmethod
    def _make_pool(iterable):
        return Pool(min(cpu_count() - 1, len(iterable), 12))

    @staticmethod
    def _save_plot(name, xlabel, ylabel):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
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
            color=mapper.to_rgba([np.mean(durations) for durations in self.durations]),
            edgecolor='black',
            ecolor='red',
            capsize=20,
        )
        plt.colorbar(mapper).set_label('seconds per turn')
        self._save_plot('tournament_ratings', None, 'TrueSkill Rating')

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
        self._save_plot('tournament_convergence', 'Match #', 'TrueSkill Rating')

        plt.figure(figsize=(16, 8))
        for player, ratings, durations in zip(self.players, self.ratings, self.durations):
            plt.errorbar(
                y=ratings[-1].mu,
                yerr=ratings[-1].sigma,
                x=np.mean(durations),
                xerr=np.std(durations),
                fmt='o',
                label=str(player),
            )
        plt.legend()
        self._save_plot('tournament_tradeoff', 'Move Duration in Seconds', 'TrueSkill Rating')
