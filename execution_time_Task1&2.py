import matplotlib
from hexgame import HexGame
from hexplayer import HexPlayerHuman, HexPlayerRandom, HexPlayerDijkstra, HexPlayerEnhanced
from hexcolour import HexColour
import sys
import os
import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()


def simulate_game(sizes, depths, iter, player1, player2):
    combinations = [(s,d) for s in sizes for d in depths]
    df = pd.DataFrame(columns=['Board size', 'Depth', 'Time'])
    for comb in combinations:
        print(comb)
        sys.stdout = open(os.devnull, 'w')
        s = comb[0]
        d = comb[1]
        for i in range(iter):
            game = HexGame(s, player1(d), player2)
            start = time.time()
            game.step([])
            df.loc[len(df)] = (s, d, time.time()-start)
        sys.stdout = sys.__stdout__

    return df


if __name__ == '__main__':
    # Players to be tested:
    # HexPlayerRandom
    # HexPlayerDijkstra
    # HexPlayerID_TT Not available

    players = [HexPlayerRandom, HexPlayerDijkstra]
    task = ['1', '2', '3']
    player2 = None

    iter = 10

    depth = range(1, 5) # range(1, 5)
    board_size = range(2, 6) # range(1, 6)

    for i in range(len(players)):
        df = simulate_game(board_size, depth, iter, players[i], player2)

        df['Board size'] = df['Board size'].astype(int)
        df['Depth'] = df['Depth'].astype(int).astype('category')
        print(df.head())

        ax = sns.lineplot(x="Board size", y="Time", hue="Depth", data=df)
        plt.yscale('log')
        plt.xticks(board_size)
        plt.ylabel('Log10 of time (s)')
        # plt.title('Execution time for a step for the hex program in task ' + task[i] + ' over ' + str(iter) + ' iterations.')
        plt.savefig('time_task' + task[i] + '.png')
        plt.clf()
