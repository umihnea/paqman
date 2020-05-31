import matplotlib.pyplot as plt
from datetime import datetime


def get_timestamp():
    now = datetime.now()
    return now.strftime('%d_%m_%Y_%H_%M_%S')


def plot_scores(scores, epsilons, output_directory):
    print(scores)
    print(epsilons)

    x = list(range(len(scores)))
    plt.subplot(1, 2, 1)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.scatter(x, scores, c='k')

    plt.subplot(1, 2, 2)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.plot(x, epsilons, c='g')

    plt.tight_layout()
    plt.savefig(output_directory + '/score_plot_' + get_timestamp() + '.png')
    plt.clf()


def plot_ram(ram_values, output):
    fig = plt.figure()
    ax = plt.axes()

    x = list(range(len(ram_values)))
    ax.plot(x, ram_values, c='b')
    plt.xlabel('Episode')
    plt.ylabel('RAM Usage (bytes)')

    fig.savefig(output + '/ram_usage_plot_' + get_timestamp() + '.png')
    plt.clf()
