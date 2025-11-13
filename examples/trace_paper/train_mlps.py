# minimal_mlp_mnist.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

datasets.MNIST.resources = [
    ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
     "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
     "d53e105ee54ea40749a09fcbcd1e9432"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
     "9fb629c4189551a2d022fa330f9573f3"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
     "ec29112dd5afa0611ce80d1b7f02629c"),
]


# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=28*28, hidden=256, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)          # flatten 28x28
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ----------------------------
# Data
# ----------------------------
def get_loaders(batch_size=128):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# ----------------------------
# Eval
# ----------------------------
@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / total

# ----------------------------
# Train one run and save init/final
# ----------------------------
def train_once(
    idx: int,
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 128,
    hidden: int = 256,
    seed: int | None = None
):
    """
    Train an MLP on MNIST once and save:
      - initial model: data/mlps_mnist/model_init_{idx}.pt
      - final model:   data/mlps_mnist/model_final_{idx}.pt
    Returns final test accuracy.
    """
    # Reproducibility per run (optional)
    if seed is None:
        seed = 1337 + idx
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs("data/mlps_mnist", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders(batch_size=batch_size)

    model = MLP(hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Save initial weights
    torch.save(model.state_dict(), f"data/mlps_mnist/model_init_{idx}.pt")

    # Train
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    # Save final weights
    torch.save(model.state_dict(), f"data/mlps_mnist/model_final_{idx}.pt")

    # Return test accuracy for convenience
    acc = eval_accuracy(model, test_loader, device)
    return acc

if __name__ == "__main__":
    results = []
    acc = train_once(idx=0, epochs=3, lr=1e-3, batch_size=128, hidden=256)
    print(f"Accuracy: {acc}")
