import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from model import SimpleCNN


def get_train_loader(batch_size: int = 128):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5)), transforms.RandomErasing(p=0.15,  # 15 % chance per image
                                                                           scale=(0.02, 0.2),
                                                                           ratio=(0.3, 3.3),
                                                                           value=0)]

    tfm = transforms.Compose(aug)
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])"""
    train_set = datasets.CIFAR10(root="data",
                                 train=True,
                                 download=True,
                                 transform=tfm)
    return DataLoader(train_set,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=2,
                      pin_memory=False)


def train(epochs, lr, batch_size, save_path, weight_decay, patience):
    device = "cpu"
    model = SimpleCNN().to(device)
    loader = get_train_loader(batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    early_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()

        val_acc = correct / total
        print(f"Epoch is {epoch} ||| "
              f"loss is {running_loss / total:.4f} ||| "
              f"accuracy is {correct / total:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            early_ctr = 0
            Path("training").mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            early_ctr += 1
            if early_ctr >= patience:
                print(f"Stopping: no val improvement for {patience} epochs.")
                break

    Path("training").mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n Save to - {save_path}")


if __name__ == "__main__":
    train(epochs=60, lr=1e-3, batch_size=128, save_path="training/model2", weight_decay=5e-4, patience=5)
