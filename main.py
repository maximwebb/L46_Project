from data import get_test_loader, get_train_loader
from models import BasicCNN

import torch
from torch import nn
import torch.optim as optim
from torch.nn.functional import softmax

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def get_class_names():
    return ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def imshow(im):
    im = im / 2 + 0.5
    npimg = im.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def top_5_classes(y):
    class_names = get_class_names()
    p = softmax(y[0, :], dim=0)
    values, indices = p.topk(5)
    return [(class_names[index], value) for index, value in zip(indices.detach().numpy(), values.detach().numpy())]


def train(net, train_data, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_data):
            inputs, labels = data

            optimizer.zero_grad()

            y = net(inputs)
            loss = criterion(y, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0


def test(model, test_data):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data:
            images, labels = data
            y = model(images)
            _, preds = torch.max(y.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    print(f"Accuracy of model: {100 * correct / total:.1f}%")


if __name__ == "__main__":
    img = Image.open("data/blas.jpg")

    train_loader = get_train_loader()
    test_loader = get_test_loader()
    basic_cnn = BasicCNN()

    PATH = "saved_models/basic_cnn.pth"
    if os.path.exists(PATH):
        print("Loading existing model")
        basic_cnn.load_state_dict(torch.load(PATH))
    else:
        train(basic_cnn, train_loader, 3)
        torch.save(basic_cnn.state_dict(), PATH)

    test(basic_cnn, test_loader)
