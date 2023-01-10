import json
import math

from data import get_test_loader, get_train_loader, get_train_loader_mnist, get_test_loader_mnist
from models import BasicCNN

import torch

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

from plot import plot_test_scores, plot_loss_curves_epsilon, plot_test_scores_with_lot_size, plot_loss_curves_lot_size, \
    plot_test_scores_with_lr
from train import train, test


def imshow(im):
    im = im / 2 + 0.5
    npimg = im.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    plot_test_scores("data/results_eps_final.json")
    plot_loss_curves_epsilon("data/results_eps_final.json", epsilons=[0.1, 0.25, 0.5, 1, 3, 5])
    plot_test_scores_with_lot_size("data/results_lot_final.json")
    plot_loss_curves_lot_size("data/results_lot_final.json", lot_sizes=[1, 2, 5, 10, 20, 100, 150])
    plot_test_scores_with_lr("data/results_lr_final.json")

    learning_rates = []
    epsilons = []
    results = []
    for learning_rate in learning_rates:
        res = {"lr": learning_rate}

        train_loader = get_train_loader_mnist(lot_size=30)
        test_loader = get_test_loader_mnist()
        # train_loader = get_train_loader()
        # test_loader = get_test_loader()
        basic_cnn = BasicCNN()

        retrain = True

        PATH = f"saved_models/dp_cnn_lr_{learning_rate}.pth"
        if os.path.exists(PATH) and not retrain:
            print("Loading existing model")
            basic_cnn.load_state_dict(torch.load(PATH))
        else:
            acc, loss = train(basic_cnn, train_loader, 10, epsilon=0.5, sensitivity=3, lot_size=30,
                              report_iterations=1000)
            print(f"LR={learning_rate} acc: {acc}")
            print(f"LR={learning_rate} loss: {loss}")
            torch.save(basic_cnn.state_dict(), PATH)
            res["acc"] = acc
            res["loss"] = loss
        score = test(basic_cnn, test_loader)
        res["score"] = score
        results.append(res)
    result_json = json.dumps({"data": results}, indent=4)
    with open("data/results_lr_bonus.json", 'w') as f:
        f.write(result_json)
        # plt.show()
