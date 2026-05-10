import torch


def get_device():
    if torch.is_vulkan_available():
        return torch.device("vulkan")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


device = get_device()

a = torch.tensor(1.0, device=device)
b = torch.tensor(2.0).to(device)

print(a, b, sep="\n")
