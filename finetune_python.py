import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ── 1. Training config ──
train_config = {
    "learning_rate": 0.001,
    "num_epochs": 10,
    "batch_size": 64,
}

start_time_sec = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 2. Data preparation ──
# Convert 1-channel 28x28 MNIST to 3-channel 224x224 for pretrained resnet18
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=train_config["batch_size"],
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=train_config["batch_size"],
    shuffle=False,
)


# ── 3. Model setup ──
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# freeze: True
for param in backbone.parameters():
    param.requires_grad = False

head = nn.Sequential(
    nn.Linear(1000, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

model = nn.Sequential(backbone, head).to(device)


# ── 4. Loss function and optimizer ──
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])


# ── 5. Training functions ──
def train_one_batch(batch):
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    logits = model(images)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_one_epoch(train_loader):
    model.train()

    total_loss = 0.0

    for batch in train_loader:
        batch_loss = train_one_batch(batch)
        total_loss += batch_loss

    return total_loss / len(train_loader)


# ── 6. Evaluation function ──
@torch.no_grad()
def evaluate(test_loader):
    model.eval()

    correct_count = 0
    total_count = 0

    for batch in test_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predicted_labels = torch.argmax(logits, dim=1)

        correct_count += (predicted_labels == labels).sum().item()
        total_count += labels.size(0)

    return {
        "accuracy_percent": correct_count * 100.0 / total_count
    }


# ── 7. Main training loop ──
epoch = 1
average_loss = 0.0

while epoch <= train_config["num_epochs"]:
    average_loss = train_one_epoch(train_loader)
    print({"epoch": epoch, "average_loss": average_loss})
    epoch += 1

print(evaluate(test_loader))

end_time_sec = time.time()
print({"elapsed_time_sec": end_time_sec - start_time_sec})