import json
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np


def plot_test_scores(file_name):
    plt.style.use('ggplot')
    with open(file_name, 'r') as f:
        results = json.load(f)["data"]
    epsilons = [result["epsilon"] for result in results if result["epsilon"] != "$\\infty$"]
    scores = [result["score"] for result in results if result["epsilon"] != "$\\infty$"]

    plt.plot(epsilons, scores)
    plt.xlabel("$\\varepsilon$")
    plt.ylabel("Test accuracy")
    plt.savefig("plots/test_acc_against_epsilon.png")
    plt.close()


def plot_test_scores_with_lot_size(file_name):
    plt.style.use('ggplot')
    with open(file_name, 'r') as f:
        results = json.load(f)["data"]
    lot_sizes = [result["lot_size"] for result in results]
    scores = [result["score"] for result in results]

    plt.plot(lot_sizes, scores)
    plt.xlabel("Lot size")
    plt.ylabel("Test accuracy")
    plt.savefig("plots/test_acc_against_lot_size.png")
    plt.close()


def plot_test_scores_with_lr(file_name):
    plt.style.use('ggplot')
    with open(file_name, 'r') as f:
        results = json.load(f)["data"]
    learning_rates = [result["lr"] for result in results]
    scores = [result["score"] for result in results]

    plt.plot(learning_rates, scores)
    plt.xlabel("Learning rate")
    plt.ylabel("Test accuracy")
    plt.savefig("plots/test_acc_against_lr.png")
    plt.close()


def plot_loss_curves_epsilon(file_name, epsilons: Optional[List[float]]):
    plt.style.use('ggplot')
    with open(file_name, 'r') as f:
        results = json.load(f)["data"]
    x = np.linspace(0, 10, len(results[0]["loss"]))
    for result in results:
        if result["epsilon"] in epsilons or result["epsilon"] == "$\\infty$":
            plt.plot(x, result["loss"], label=f"$\\varepsilon$={result['epsilon']}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/loss_against_epoch_epsilon_curves.png")
    plt.close()


def plot_loss_curves_lot_size(file_name, lot_sizes: Optional[List[float]]):
    plt.style.use('ggplot')
    with open(file_name, 'r') as f:
        results = json.load(f)["data"]
    for result in results:
        x = np.linspace(0, 10, len(result["loss"]))
        if result["lot_size"] in lot_sizes:
            plt.plot(x, result["loss"], label=f"L={result['lot_size']}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/loss_against_epoch_lot_size_curves.png")
    plt.close()
