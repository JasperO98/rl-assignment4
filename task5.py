from hexplayer import HexPlayerDijkstra, HexPlayerRandom, HexPlayerHuman
from hexgame import HexGame
from trueskill import rate_1vs1
import matplotlib.pyplot as plt


#How many games should be played for the rating to stabilize statistically?
#Can you calculate this number?
#How fast is your program, what board size will you use?



def VC_bar_plot(log,player1,player2):
    y = [log[0][-1],log[1][-1]]
    x = [player1,player2]
    plt.bar(x,y)
    plt.xlabel('Players')
    plt.ylabel('Skill (mu)')
    plt.title('Skill values of all users')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(y):
        plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v,2)))
    plt.xticks(xlocs, x)
    plt.show()


# def new_users(player1, player2):
#     if player1[0] == "dijkstra":
#         player1_obj = HexPlayerDijkstra(player1[1])
#     else:
#         player1_obj = HexPlayerRandom(player1[1])
#     if player2[0] == "dijkstra":
#         player2_obj = HexPlayerDijkstra(player2[1])
#     else:
#         player2_obj = HexPlayerRandom(player2[1])
#     return player1_obj, player2_obj


def match(player1, player2, n_games, size):
    log_player1 =[]
    log_player2 = []
    for i in range(n_games):
        game = HexGame(size, player1, player2)
        game.play([])
        print(game.win)
        if game.win[1] == 1: # player 1 win
            player1.rating, player2.rating = rate_1vs1(player1.rating, player2.rating, drawn=False)
        elif game.win[1] == 2: # player 2 win
            player2.rating, player1.rating = rate_1vs1(player2.rating, player1.rating, drawn=False)
        log_player1.append(player1.rating.mu -3 * player1.rating.sigma)
        log_player2.append(player2.rating.mu - 3 * player2.rating.sigma)
    return [log_player1, log_player2]

def main():
    n_runs = 1
    board_size = 4

    ## run1; dijkstra3 vs dijkstra4
    print("dijkstra3 vs dijkstra4")
    log = match(HexPlayerDijkstra(3), HexPlayerDijkstra(4), n_runs, board_size)
    VC_bar_plot(log,"dijkstra3","dijkstra4")
    print("\n")

   #  ## run2; dijkstra3 vs random3
    print("dijkstra3 vs random3")
    log = match(HexPlayerDijkstra(3), HexPlayerRandom(3), n_runs, board_size)
    VC_bar_plot(log,"dijkstra3","random")
    print("\n")


   #  ## run3; dijkstra4 vs random3
    print("dijkstra4 vs random3")
    log = match(HexPlayerDijkstra(4), HexPlayerRandom(3), n_runs, board_size)
    VC_bar_plot(log,"dijkstra4","random3")
    print("\n")
    print(log)

if __name__ == "__main__":
    main()