import torch

if torch.xpu.is_available():
    print("Intel GPU is Available!\n")
    device = torch.device("xpu")
    xpu_name = torch.xpu.get_device_name(device=device)
    print(f"NAME: {xpu_name}\n")
    xpu_props = torch.xpu.get_device_properties(device=device)
    xpu_capability = torch.xpu.get_device_capability(device)
    print(f"PROPERTIES:\n{xpu_props}\n")
    print(f"CAPABILIES:\n{xpu_capability}\n")
else:
    print("No Intel GPU is Available!")

