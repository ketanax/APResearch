# Import the libraries
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def test(model, device, test_loader, criterion):

    model.eval()

    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            # Top-1
            _, preds = torch.max(outputs, 1)
            correct_top1 += (preds == labels).sum().item()

            # Top-5
            _, top5_preds = torch.topk(outputs, 5, dim=1)
            correct_top5 += (top5_preds == labels.unsqueeze(1)).sum().item()

            total += labels.size(0)

    test_loss = running_loss / total
    top1_acc = 100 * correct_top1 / total
    top5_acc = 100 * correct_top5 / total

    return test_loss, top1_acc, top5_acc