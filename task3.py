import setup
from hex.tournament import HexTournament
from hex.players.selfplay import AlphaZeroSelfPlay1
from hex.players.base import HexPlayerRandom
from hex.players.montecarlo import HexPlayerMonteCarloTime

if __name__ == '__main__':
    ht = HexTournament(5, (
        HexPlayerRandom(),
        AlphaZeroSelfPlay1(epochs=10, cp=5, episodes=50, threshold=0.5),
        AlphaZeroSelfPlay1(epochs=10, cp=5, episodes=100, threshold=0.51),
        AlphaZeroSelfPlay1(epochs=1, cp=5, episodes=50, threshold=0.5),
        HexPlayerMonteCarloTime(10, 1),
        HexPlayerMonteCarloTime(30, 1),
    ))
    ht.tournament()
    ht.plots()
