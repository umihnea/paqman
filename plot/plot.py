import matplotlib.pyplot as plt
from datetime import datetime


def get_timestamp():
    now = datetime.now()
    return now.strftime('%d_%m_%Y_%H_%M_%S')


def plot_to_file(scores, epsilons, total_episodes, output_directory):
    x = list(range(total_episodes))
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
