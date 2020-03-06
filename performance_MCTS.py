import matplotlib
import sys
import os
import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
import numpy as np

from hex.game import HexGame
from hex.players.montecarlo import HexPlayerMonteCarlo


def simulate_game(sizes, iter, player1, player2, N, Cp):
    combinations = [(s, n, c) for s in sizes for n in N for c in Cp]
    df = pd.DataFrame(columns=['Board size', 'N_Cp', 'Time'])
    for comb in combinations:
        # print(comb)
        # sys.stdout = open(os.devnull, 'w')
        s = comb[0]
        n = comb[1]
        c = comb[2]
        for i in range(iter):
            p1 = player1(n, c)
            game = HexGame(s, p1, player2)
            start = time.time()
            game.step([])
            df.loc[len(df)] = (s, str(n) + '_' + str(c), time.time()-start)
        # sys.stdout = sys.__stdout__
    return df


if __name__ == '__main__':
    N = [100]
    Cp = [1]
    board_size = range(3, 7)

    task = 'MCTS_N{}-{}_Cp{}-{}_Size{}-{}'.format(min(N), max(N), min(Cp), max(Cp), min(board_size), max(board_size))

    player1 = HexPlayerMonteCarlo
    player2 = None

    iter = 1

    df = simulate_game(board_size, iter, player1, player2, N, Cp)

    df['Board size'] = df['Board size'].astype(int).astype('category')
    df['N_Cp'] = df['N_Cp'].astype('category')

    print(df.head())

    # plt.yscale('log')

    ax = sns.lineplot(x="Board size", y="Time", hue="N_Cp", data=df)

    plt.xticks(board_size)

    plt.xlabel('Board size')
    plt.ylabel('Time (s)')

    plt.savefig('time_task' + task + '.png')
    plt.clf()