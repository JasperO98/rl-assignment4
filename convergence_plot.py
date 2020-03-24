from  trueskill import TrueSkill, rate_1vs1
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


ENV = TrueSkill(mu=25, sigma=8 + 1 / 3, draw_probability=0)

p1 = ENV.create_rating()
p2 = ENV.create_rating()
p1_log = []
p2_log = []

for _ in range(1000):
    p1, p2 = rate_1vs1(p1, p2, drawn=False)
    p1_log.append(p1.mu - 3 * p1.sigma)
    p2_log.append(p2.mu - 3 * p2.sigma)
print(p1)


def line_plot(numbers):
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]
    legend = []
    player = 1
    for i in numbers:
        color_line = colors.pop()
        plt.plot(i, color=color_line, linewidth=2)
        color_patch = mpatches.Patch(color = color_line, label = "player"+str(player))
        legend.append(color_patch)
        player+=1
    plt.legend(handles = legend)
    plt.ylabel('TrueSkill Value')
    plt.xlabel('Number of iterations')
    plt.title("Covergence of skill over iterations")
    plt.show()

line_plot([p1_log, p2_log])