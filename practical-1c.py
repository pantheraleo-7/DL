import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


device = get_device()

a = torch.tensor(1.0, device=device)
b = torch.tensor(2.0).to(device)
