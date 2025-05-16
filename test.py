import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN


MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]
def get_test_loader(batch_size):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    test_set = datasets.CIFAR10(root="data",
                                train=False,
                                download=True,
                                transform=tfm)
    return DataLoader(test_set,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=2,
                      pin_memory=False)


def evaluate(model_path):

    device = "cpu"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = get_test_loader(batch_size=256)
    correct, total = 0, 0

    print(total)

    with torch.no_grad():  # so we don't have to reset the gradient
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    print(f"Test accuracy is {correct / total:.4f}")


if __name__ == "__main__":
    evaluate("training/modelIter5")
