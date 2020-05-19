import setup
from hex.tournament import HexTournament
from hex.players.selfplay import AlphaZeroSelfPlay1, AlphaZeroSelfPlay2
from glob import glob
from os.path import join, sep

if __name__ == '__main__':
    players = []

    for path in glob(join('models', '5x5', '*', 'player1')):
        players.append(AlphaZeroSelfPlay1(int(path.split(sep)[-2])))
    for path in glob(join('models', '5x5', '*', 'player2')):
        players.append(AlphaZeroSelfPlay2(int(path.split(sep)[-2])))

    ht = HexTournament(5, players)
    ht.train()
    ht.tournament()
    ht.plots()
