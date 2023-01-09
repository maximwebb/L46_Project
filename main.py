from data import get_test_loader, get_train_loader, get_train_loader_mnist, get_test_loader_mnist
from models import BasicCNN

import torch

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

from train import train, test


def imshow(im):
    im = im / 2 + 0.5
    npimg = im.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    img = Image.open("data/blas.jpg")

    batch_size = 10

    train_loader = get_train_loader_mnist(batch_size=batch_size)
    test_loader = get_test_loader_mnist()
    # train_loader = get_train_loader()
    # test_loader = get_test_loader()
    basic_cnn = BasicCNN()

    retrain = False

    # PATH = "asdf"
    PATH = "saved_models/dp_cnn_actual.pth"
    if os.path.exists(PATH) and not retrain:
        print("Loading existing model")
        basic_cnn.load_state_dict(torch.load(PATH))
    else:
        acc, loss = train(basic_cnn, train_loader, 8, epsilon=4, sensitivity=3, lot_size=batch_size)
        print(acc)
        torch.save(basic_cnn.state_dict(), PATH)
        x = np.linspace(0, 2000 * len(acc), len(acc))
        plt.plot(x, acc, label="Accuracy")
        plt.plot(x, loss, label="Loss")
        plt.legend()

    test(basic_cnn, test_loader)
    plt.show()
