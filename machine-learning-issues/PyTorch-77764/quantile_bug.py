"""
"""

import torch

device = torch.device("mps")

A = torch.randn(1000, 10)

# need to use "nearest", "lower", or "higher" interpolation methods
# "linear" and "midpoint" fail due to operator `aten::lerp.Tensor_out` not being implemented for MPS
# when dim=None, CPU and MPS give same result
print("TEST: dim=None, CPU and MPS give same result")
if torch.equal(torch.quantile(A, .5, dim=None, interpolation = "nearest"), 
                   torch.quantile(A.to(device), .5, dim=None, interpolation = "nearest").to("cpu")):
    print("Success: MPS and CPU give the same result when dim=None")
else:
    print("Error: MPS and CPU give different result when dim=None")


# when dim is not None, MPS gives nonsense result
print("TEST: dim=0, CPU and MPS give different result")
if torch.equal(torch.quantile(A, .5, dim=0, interpolation = "nearest"),
                   torch.quantile(A.to(device), .5, dim=0, interpolation = "nearest").to("cpu")):
    print("Success: MPS and CPU give the same result when dim=0")
else:
    print("Error: MPS and CPU give different result when dim=0")



# inelegant workaround is to sort the tensor before calling quantile (this is not a good solution for large tensors)
print("TEST: dim=0, sorted tensor, CPU and MPS give same result")
if torch.equal(torch.quantile(torch.sort(A, dim = 0).values, .5, 0, interpolation = "nearest"), 
                   torch.quantile(torch.sort(A.to(device), dim = 0).values, .5, 0, interpolation = "nearest").to("cpu")):
    print("Success: MPS and CPU give the same result with sorted tensor")
else:
    print("Error: MPS and CPU give different result with sorted tensor")