import math
from typing import Optional

import numpy as np
import torch
from torch import nn, optim


def laplace_mechanism(grad, epsilon, sensitivity, lot_size):
    noise = torch.from_numpy(np.random.laplace(0, scale=sensitivity / (epsilon * lot_size), size=grad.size())).float()
    norm = np.linalg.norm(np.reshape(grad, -1), ord=2)
    clipped_grad = grad if norm < sensitivity else sensitivity * grad / norm
    return clipped_grad + noise


def train(net, train_data, epochs, epsilon: Optional[float], sensitivity: Optional[float], lot_size=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Apply Laplace Mechanism by registering hook
    if epsilon and sensitivity:
        for p in net.parameters():
            p.register_hook(lambda grad: laplace_mechanism(grad, epsilon, sensitivity, lot_size))

    loss_scores = []
    accuracy_scores = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct_count = 0
        total_count = 0
        for i, data in enumerate(train_data):
            inputs, labels = data

            optimizer.zero_grad()

            y = net(inputs)
            loss = criterion(y, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(y.data, 1)
            total_count += labels.size(0)
            correct_count += (preds == labels).sum().item()
            if i % 2000 == 1999:
                print(
                    f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}, '
                    f'acc: {correct_count / total_count:.3f}')
                # for j, layer in enumerate(net.children()):
                #     if hasattr(layer, "weight"):
                #         print(f"Layer {j}: {layer.weight.size()}, {np.linalg.norm(np.reshape(layer.weight.grad, -1), ord=2)}")
                loss_scores.append(running_loss / 2000)
                accuracy_scores.append(correct_count / total_count)
                correct_count = 0
                running_loss = 0.0
                total_count = 0
    print(f"Accuracy: {accuracy_scores}")
    print(f"Loss: {loss_scores}")
    return accuracy_scores, loss_scores


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
