# Import the libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import dataset as ds
from train import train
from test import test
from model import ResNet50

BASE_DIR = "output"
MODEL_DIR = os.path.join(BASE_DIR, "ResNet")

os.makedirs(MODEL_DIR, exist_ok=True)

LOG_FILE = os.path.join(MODEL_DIR, "training_log.txt")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

# Import the datasets and create the dataloaders
train_ds = ds.train_dataset
test_ds = ds.test_dataset

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)


# Define the model + weight initialitzation
# (used Xavier uniform, even though in the paper they use random initialization or weights of previously trained shallow networks)
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

num_classes = 1000
model = ResNet50(num_classes).to(device)  # (input_dim, num_classes)
model.apply(initialize_weights)

# Set hyperparameters
EPOCHS = 70
lr = 0.01

# Optimitzer and loss
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for e in range(EPOCHS):
    loss_train = train(model, device, train_loader, optimizer, e, criterion)
    loss_test, accur_test = test(model, device, test_loader, criterion)
    lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    print('Epoch: {} \tTrainLoss: {:.6f}\tValidationLoss: {:.6f}\tAccuracy validation: {:.6f}'.format(
        e,
        loss_train,
        loss_test,
        accur_test
    ))

    with open(LOG_FILE, "a") as f:
        f.write(f"{e},{loss_train},{loss_test},{accur_test}\n")

    # Save model each epoch
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, f"ResNet_epoch_{e}.pth"))