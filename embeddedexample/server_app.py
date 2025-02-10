from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        return self.fc3(x)


def get_weights(net):
    """Extract weights from the model."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from provided parameters."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def load_data_from_disk(path: str, batch_size: int):
    """Load dataset in Hugging Face format and create PyTorch dataloaders."""
    partition_train_test = load_from_disk(path)

    # Define image transformations
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    def apply_transforms(batch):
        """Apply transformations to dataset partitions."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=batch_size)

    return trainloader, valloader


def train(net, trainloader, valloader, epochs, learning_rate, device):
    """Train the model using the provided dataset."""
    net.to(device)  # Move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            criterion(net(images), labels).backward()
            optimizer.step()

    val_loss, val_acc = test(net, valloader, device)

    results = {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, testloader, device):
    """Evaluate the model using the validation dataset."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    return loss, accuracy


(my-env) server@raspberrypi:~/embedded-devices $ cat embeddedexample/server_app.py
"""embeddedexample: A Flower / PyTorch app with XFL strategy."""

from typing import List, Tuple
import flwr as fl
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from embeddedexample.task import Net, get_weights
from embeddedexample.xfl_strategy import XFLStrategy

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    num_rounds = context.run_config["num-server-rounds"]
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = XFLStrategy(
        fraction_fit=1.0,
        min_fit_clients=4,
        min_available_clients=4
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)