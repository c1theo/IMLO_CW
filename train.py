import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from model import SimpleCNN
import time

# CIFAR-10 statistics
MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]
MAX_DURATION = 3 * 3600 + 50 * 60 # Does not let model train for more than 3 hours 50
def get_train_loader(batch_size = 128):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
           transforms.ColorJitter(
               brightness=0.2,
               contrast=0.2,
               saturation=0.2,
               hue=0.1),
           transforms.ToTensor(), transforms.Normalize(MEAN, STD),
           transforms.RandomErasing(p=0.20,  # 20 % chance per image - change pixel area to mean colour
                                    scale=(0.02, 0.2),
                                    ratio=(0.3, 3.3), value=MEAN),
           transforms.RandomErasing(p=0.10,  # 10 % chance per image - change pixel area to black
                                    scale=(0.05, 0.15), # slightly more consistent size
                                    ratio=(0.3, 3.3), value=MEAN)
           ]
    # changed pipeline order since last execution - was normalizing before random erasing
    # changes pixels to MEAN so it looks normal rather than just black and have added another random erasing
    # with smaller probality for just black

    tfm = transforms.Compose(aug)
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
    start_time = time.time()
    device = "cpu"
    model = SimpleCNN().to(device)
    loader = get_train_loader(batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    early_ctr = 0

    for epoch in range(1, epochs + 1):
        elapsed = time.time() - start_time
        if elapsed >= MAX_DURATION:
            print(f"Stopped early - reached near maximum allowed time")
            break
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
        val_loss = running_loss / total
        scheduler.step(val_loss)

        val_acc = correct / total
        elapsed_time = time.time() - start_time
        print(f"Epoch is {epoch} ||| "
              f"loss is {val_loss:.4f} ||| "
              f"accuracy is {correct / total:.4f} ||| "
              f"current time {elapsed_time/60:.2f} min")

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
    train(epochs=300, lr=1e-3, batch_size=128, save_path="training/modelIter8", weight_decay=5e-4, patience=10)