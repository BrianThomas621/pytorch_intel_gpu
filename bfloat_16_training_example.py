# https://intel.github.io/intel-extension-for-pytorch/xpu/2.7.10+xpu/tutorials/examples.html
import torch
import torchvision
import intel_extension_for_pytorch as ipex

if torch.xpu.is_available():
    print("Intel GPU is Available!\n")
    device = torch.device("xpu")
    xpu_name = torch.xpu.get_device_name(device=device)
    print(f"NAME: {xpu_name}\n")

LR = 0.001
DOWNLOAD = True
DATA = "datasets/cifar10/"

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA,
    train=True,
    transform=transform,
    download=DOWNLOAD,
)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128)

model = torchvision.models.resnet50()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
model.train()

model = model.to("xpu")
criterion = criterion.to("xpu")
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)

for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    data = data.to("xpu")
    target = target.to("xpu")
    with torch.autocast(
        device_type="xpu",
        dtype=torch.bfloat16,
        enabled=True,
    ):
        output = model(data)
        loss = criterion(output, target)

    loss.backward()
    optimizer.step()
    print(f"BATCH ID: {batch_idx}")


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "checkpoint.pth",
)

print("Execution finished")
