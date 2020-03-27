from hex.players.alphabeta import HexPlayerEnhancedAB
from hex.players.base import HexPlayerRandom
from hex.tournament import HexTournament
from hex.players.montecarlo import HexPlayerMonteCarloTime, HexPlayerMonteCarloIterations
from  trueskill import TrueSkill, rate_1vs1
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches





def convergence():
    ENV = TrueSkill(mu=25, sigma=8 + 1 / 3, draw_probability=0)
    p1 = ENV.create_rating()
    p2 = ENV.create_rating()
    p1_log = []
    p2_log = []
    for _ in range(400):
        p1, p2 = rate_1vs1(p1, p2, drawn=False)
        p1_log.append(p1.mu - 3 * p1.sigma)
        p2_log.append(p2.mu - 3 * p2.sigma)
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]
    legend = []
    player = 1
    for i in [p1_log, p2_log]:
        color_line = colors.pop()
        plt.plot(i, color=color_line, linewidth=2)
        color_patch = mpatches.Patch(color = color_line, label = "player"+str(player))
        legend.append(color_patch)
        player+=1
    plt.ylabel('TrueSkill Value')
    plt.xlabel('Number of games played')
    plt.title("Covergence of trueskill value for each player")
    plt.legend(handles = legend)
    plt.savefig("figures/perfect_convergence.pdf")
    plt.close()


if __name__ == '__main__':
    ht = HexTournament(4, (HexPlayerRandom(),
                           HexPlayerMonteCarloTime(10, 0.5),
                           HexPlayerMonteCarloTime(10, 1.0),
                           HexPlayerMonteCarloTime(10, 1.5),
                           HexPlayerMonteCarloTime(10, 2.0),
                           HexPlayerEnhancedAB(15, True),
                           HexPlayerEnhancedAB(10, True),
                           HexPlayerEnhancedAB(5, True),
                           HexPlayerEnhancedAB(2, True)
                        ))
    ht.tournament()
    ht.task3()
    
    ht_convergence1 = HexTournament(4, (HexPlayerMonteCarloTime(10, 0.5),
                                       HexPlayerEnhancedAB(15, True)))
    ht_convergence1.convergence('figures/plot_MTCT0_5_IDTT_15.pdf',
                                'figures/bar_MTCT0_5_IDTT_15.pdf')
    
    
    ht_convergence2 = HexTournament(4, (HexPlayerEnhancedAB(10, True),
                                       HexPlayerEnhancedAB(15, True)))
    ht_convergence2.convergence('figures/plot_IDTT_15_IDTT_10.pdf',
                                'figures/bar_IDTT_15_IDTT_10.pdf')
    
        
    ht_convergence3 = HexTournament(4, (HexPlayerMonteCarloTime(10, 0.5),
                                       HexPlayerEnhancedAB(2, True)))
    ht_convergence3.convergence('figures/plot_MTCT0_5_IDTT_2.pdf',
                                'figures/bar_MTCT0_5_IDTT_2.pdf') 
    
    ht_convergence4 = HexTournament(4, (HexPlayerMonteCarloTime(10, 0.5),
                                       HexPlayerMonteCarloTime(10, 2.0)))
    ht_convergence4.convergence('figures/plot_MTCT0_5__MTCT_2.pdf',
                                'figures/bar_MTCT0_5__MTCT_2.pdf')
    convergence()
    