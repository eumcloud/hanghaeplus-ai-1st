import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.optim import SGD
from matplotlib import pyplot as plt

# Dataset and dataloader
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

# Simple Model for Testing
class Model(nn.Module):
    def __init__(self, input_dim, n_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, n_dim)
        self.layer2 = nn.Linear(n_dim, n_dim)
        self.layer3 = nn.Linear(n_dim, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        return x

# Instantiate model and optimizer
model = Model(28 * 28 * 1, 1024)
optimizer = SGD(model.parameters(), lr=0.001)

# Simple Test Loop for 1 Epoch
for epoch in range(1):
    total_loss = 0.
    for data in trainloader:
        inputs, labels = data
        optimizer.zero_grad()
        preds = model(inputs)
        loss = (preds[:, 0] - labels).pow(2).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss}")
