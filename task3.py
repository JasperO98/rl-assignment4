from hex.players.alphabeta import HexPlayerEnhancedAB
from hex.players.base import HexPlayerRandom
from hex.tournament import HexTournament
import matplotlib.pyplot as plt
from hex.players.montecarlo import HexPlayerMonteCarloTime, HexPlayerMonteCarloIterations
import pickle


def bar_plot(ratings, names, plot_title=''):
    y = ratings
    x = names
    plt.bar(x, y, align='center', alpha=0.5)
    plt.ylabel('TrueSkill value')
    plt.title(plot_title)
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(y):
        plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
    plt.xticks(xlocs, x)
    plt.show()

#    plt.savefig('figures/ratings.pdf')

if __name__ == '__main__':
    ht = HexTournament(4, (HexPlayerRandom(),
                           HexPlayerRandom(),
                           HexPlayerRandom()
                           ))
    print(ht.players)
    print(ht.ratings)
    print(ht.durations)
    print(ht.ratings_log)
    print("--------------\n" * 5)
    ht.tournament()
    ht.task3()
    print("--------------\n" * 5)
    print(ht.players)
    print(ht.ratings)
    print(ht.durations)
    print(ht.ratings_log)


# main()
