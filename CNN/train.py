# Import the libraries
from tqdm import tqdm

def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    running_loss = 0.0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 🔹 print every 50 iterations
        if (batch_idx + 1) % 50 == 0:
            avg_loss = running_loss / 50
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
            running_loss = 0.0

    return loss.item()
