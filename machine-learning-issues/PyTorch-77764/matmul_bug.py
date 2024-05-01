import torch

zeros = torch.zeros(911, 9, 1, device=torch.device("mps"))
ones = torch.ones(1, 32769, device=torch.device("mps"))
zeros @ ones