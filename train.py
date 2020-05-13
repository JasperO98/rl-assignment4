import setup
from hex.players.selfplay import AlphaZeroSelfPlay1, AlphaZeroSelfPlay2

if __name__ == '__main__':
    settings = dict(epochs=10, cp=5, episodes=50, threshold=0.5)
    AlphaZeroSelfPlay1(**settings).setup(5)
    AlphaZeroSelfPlay2(**settings).setup(5)
