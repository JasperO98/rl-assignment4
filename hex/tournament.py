from math import log, ceil
from hex.game import HexGame
from hex.colour import HexColour
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import permutations
from hex.players.montecarlo import HexPlayerMonteCarloTime
from trueskill import rate_1vs1
import matplotlib.pyplot as plt
import seaborn as sns
from time import time


class HexTournament:
    def __init__(self, size, players):
        self.size = size
        self.players = players
        self.durations = [0] * len(players)

    def _match_unpack(self, args):
        return self.match(*args)

    def match(self, pi1, pi2):
        start = time()
        winner = HexGame(self.size, self.players[pi1], self.players[pi2]).play([])
        stop = time()

        if winner == HexColour.RED:
            return pi1, pi2, stop - start
        elif winner == HexColour.BLUE:
            return pi2, pi1, stop - start

    def tournament(self):
        matches = list(permutations(range(len(self.players)), 2)) * ceil(2 * log(len(self.players), 2))
        with Pool(cpu_count() - 1) as pool:
            for winner, loser, duration in tqdm(iterable=pool.imap(self._match_unpack, matches), total=len(matches)):
                self.players[winner].rating, self.players[loser].rating = rate_1vs1(self.players[winner].rating, self.players[loser].rating)
                self.durations[winner] += duration / sum(winner in match for match in matches) / 2
                self.durations[loser] += duration / sum(loser in match for match in matches) / 2

    def task3(self):
        sns.barplot(
            x=[str(player) for player in self.players],
            y=[player.rating.mu - 3 * player.rating.sigma for player in self.players],
        )
        plt.ylabel('TrueSkill Score')
        plt.show()

    def task4(self):
        pass


if __name__ == '__main__':
    ht = HexTournament(4, (
        HexPlayerMonteCarloTime(1, 1),
        HexPlayerMonteCarloTime(2, 1),
        HexPlayerMonteCarloTime(3, 1),
    ))
    ht.tournament()
    ht.task3()
    print(ht.durations)
