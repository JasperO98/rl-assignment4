from math import log, ceil
from hex.game import HexGame
from hex.colour import HexColour
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import permutations
from hex.players.montecarlo import HexPlayerMonteCarloTime, HexPlayerMonteCarloIterations
from trueskill import rate_1vs1, TrueSkill


class HexTournament:
    def __init__(self, size, players):
        self.size = size
        self.players = players
        self.durations = [0 for _ in range(len(players))]
        self.ratings = [TrueSkill(mu=25, sigma=8 + 1 / 3, draw_probability=0).create_rating() for _ in range(len(players))]
        self.history = [[] for _ in range(len(players))]
        for i in range(len(players)):
            self._add_to_history(i)

    def _add_to_history(self, index):
        self.history[index].append(self.ratings[index].mu - 3 * self.ratings[index].sigma)

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
                self._add_to_history(wi)
                self._add_to_history(li)
