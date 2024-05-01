"""
PyTorch-77764: Incorrect gradients for torch.quantile on MPS

Expected behavior: MPS and CPU should return the same gradients for torch.quantile
Actual behavior: MPS and CPU return different gradients for torch.quantile
"""

import torch

device = torch.device("mps")


A = torch.randn(1000, 10, requires_grad = True)
#CPU: correctly returns non-zero gradients for 1 element in each column of A
out = torch.sum(torch.quantile(A, .5, 0, interpolation = "nearest"))
out.backward()
cpu_grad = torch.where(A.grad != 0)

A.grad = None

#MPS: usually returns no non-zero elements
#some seeds give non-zero elements but in the wrong places (always in the first row)
out = torch.sum(torch.quantile(A.to(device), .5, 0, interpolation = "nearest"))
out.backward()
device_grad = torch.where(A.grad != 0)

for i in range(len(cpu_grad)):
    if not torch.equal(cpu_grad[i], device_grad[i]):
        print("cpu:", cpu_grad)
        print(f"{device}:", device_grad)
        raise Exception("MPS and CPU gradients are different")
print("Success: MPS and CPU gradients are the same")