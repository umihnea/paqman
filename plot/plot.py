import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np


def get_timestamp():
    now = datetime.now()
    return now.strftime("%d_%m_%Y_%H_%M_%S")


def plot_scores(scores, epsilons, output_directory):
    plt.subplot(1, 2, 1)

    smoothing_window = 5
    rewards_smoothed = (
        pd.Series(scores).rolling(smoothing_window, min_periods=smoothing_window).mean()
    )
    plt.plot(rewards_smoothed)

    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.subplot(1, 2, 2)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.plot(epsilons, c="g")

    plt.tight_layout()
    plt.savefig(output_directory + "/score_plot_" + get_timestamp() + ".png")
    plt.clf()


def plot_evaluation(scores, output):
    fig = plt.figure()
    ax = plt.axes()

    ax.plot(scores, c="b")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    mean = np.mean(scores).item()
    ax.axhline(y=mean, color="r", linestyle="-")

    fig.savefig(output + "/eval_plot_" + get_timestamp() + ".png")
    plt.clf()
