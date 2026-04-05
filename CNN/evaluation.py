import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import test_dataset
from model import ResNet50
from test import test

BATCH_SIZE = 64
NUM_WORKERS = 0
MODEL_PATH = r"C:\Users\ErogluPC\PycharmProjects\alexnet-pytorch\CNN\trained_models\CNN2.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    print("Using device:", device)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    num_classes = 1000
    print("Number of classes:", num_classes)

    model = ResNet50(num_classes=num_classes).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)

    criterion = nn.CrossEntropyLoss()

    test_loss, top1_acc, top5_acc = test(
        model=model,
        device=device,
        test_loader=test_loader,
        criterion=criterion
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")


if __name__ == "__main__":
    main()