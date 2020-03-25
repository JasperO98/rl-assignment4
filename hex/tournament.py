from math import log, ceil, floor
from hex.game import HexGame
from hex.colour import HexColour
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import permutations
from hex.players.montecarlo import HexPlayerMonteCarloTime, HexPlayerMonteCarloIterations
from trueskill import rate_1vs1, TrueSkill
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np


class HexTournament:
    def __init__(self, size, players):
        self.size = size
        self.players = players
        self.durations = [0 for _ in range(len(players))]
        self.ratings = [TrueSkill(mu=25, sigma=8 + 1 / 3, draw_probability=0).create_rating() for _ in range(len(players))]

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
                self.ratings[wi], self.ratings[li] = rate_1vs1(self.ratings[wi], self.ratings[li])
                self.durations[wi] += wd / sum(wi in match for match in matches)
                self.durations[li] += ld / sum(li in match for match in matches)

    def task3(self):
        sns.barplot(
            x=[str(player) for player in self.players],
            y=[rating.mu - 3 * rating.sigma for rating in self.ratings],
        )
        plt.ylabel('TrueSkill Score')
        plt.show()

    def task4(self):
        names = [str(player) for player in self.players]
        ratings = [rating.mu - 3 * rating.sigma for rating in self.ratings]

        plt.clf()
        plt.figure(figsize=(20, 15))
        y = ratings
        x = self.durations

        m = ['x', 's', '*', 'p', 'D', '^', 'o']
        for i in range(len(names)):
            plt.plot(x[i], y[i], linestyle='none', marker=m[int(i*0.1)], label=names[i])

        plt.subplots_adjust(left=0.05, right=0.7)
        plt.legend(numpoints=1, loc='center right', bbox_to_anchor=(1.45, 0.5), ncol=2, prop={'size': 16})

        plt.ylabel('TrueSkill value', fontsize=18)
        plt.xlabel('Average game duration in seconds', fontsize=18)

        plt.xticks(range(floor(min(x)), ceil(max(x)), 10), fontsize=16, rotation=60)
        plt.yticks(fontsize=16)


        if __name__ == '__main__':
            plt.show()
        else:
            plt.savefig('figures/task4.pdf')

            rows = [ratings, [n[5:] for n in names], x]
            with open('figures/test.csv', "w", encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row)


if __name__ == '__main__':
    ht = HexTournament(1, (
        HexPlayerMonteCarloTime(1, 1),
        HexPlayerMonteCarloTime(2, 1),
        HexPlayerMonteCarloTime(3, 1),
    ))
    ht.tournament()
    ht.task3()
    print(ht.durations)

    ht = HexTournament(1, (
        HexPlayerMonteCarloIterations(5, 1),
        HexPlayerMonteCarloIterations(5, 1),
        HexPlayerMonteCarloIterations(5, 1),
    ))
    ht.tournament()
    ht.task4()
    print(ht.durations)
