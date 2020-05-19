import setup
from hex.tournament import HexTournament
from hex.players.montecarlo import HexPlayerMonteCarloTime
from hex.players.alphabeta import HexPlayerEnhancedAB
from hex.players.base import HexPlayerRandom, HexPlayerHuman
from hex.players.selfplay import AlphaZeroSelfPlay1, AlphaZeroSelfPlay2

if __name__ == '__main__':
    players = [
        HexPlayerHuman(),
        HexPlayerRandom(),
        HexPlayerMonteCarloTime(10, 1),
        HexPlayerEnhancedAB(10, True),
        AlphaZeroSelfPlay1(),
        AlphaZeroSelfPlay2(),
    ]

    ht = HexTournament(7, players)
    ht.train()
    ht.tournament()
    ht.plots()
