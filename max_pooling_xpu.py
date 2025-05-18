import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
from time import perf_counter

import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
    xpu_name = torch.xpu.get_device_name(device=device)
print(f"Running on device: {device} : {xpu_name}")

mnist_train = datasets.FashionMNIST(
    root="./data", download=True, train=True, transform=ToTensor()
)
mnist_test = datasets.FashionMNIST(
    root="./data", download=True, train=False, transform=ToTensor()
)

train_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=(3, 3), padding=1, padding_mode="reflect"),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(588, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Begin Training Time
start_time = perf_counter()
for i in range(0, 30):
    model.train()
    loss_sum = 0
    for X, y in train_dataloader:
        y = F.one_hot(y, num_classes=10).type(torch.float32).to(device)
        X = X.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
    print(f"LOSS: {loss_sum}")

# End Training Time
end_time = perf_counter()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} Seconds")

start_eval_time = perf_counter()
model.eval()
with torch.no_grad():
    accurate = 0
    total = 0
    for X, y in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        outputs = nn.functional.softmax(model(X), dim=1)
        correct_pred = y == outputs.max(dim=1).indices
        total += correct_pred.size(0)
        accurate += correct_pred.type(torch.int).sum().item()
    print("Accuracy on Validation Data:", accurate / total)

end_eval_time = perf_counter()
eval_time = end_eval_time - start_eval_time
print(f"Evaluation Time: {eval_time:.2f} Seconds")
