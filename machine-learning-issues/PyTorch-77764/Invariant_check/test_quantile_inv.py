import torch

def invariant_check(input_tensor, dim=None, q=0.5):
    # Calculate the expected result using a reference implementation
    reference_result = torch.quantile(input_tensor, q=q, dim=dim, interpolation="nearest")
    
    # Run torch.quantile on the MPS device
    mps_result = torch.quantile(input_tensor.to('mps'), q=q, dim=dim, interpolation='nearest').to('cpu')
    
    # Check if the MPS result matches the reference result within a tolerance
    tolerance = 1e-6  # Adjust the tolerance as needed, floating point errors are expected due to different platforms
    if torch.allclose(mps_result, reference_result, rtol=tolerance, atol=tolerance):
        return True
    else:
        return False

# Example usage:
# input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]) # sorted pass
# input_tensor = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0]) # unsorted pass

input_tensor = torch.tensor([[1.0, 3.0, 2.0, 5.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]]) # unsorted fail
# input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]) # sorted pass
# input_tensor = torch.randn(1000, 10) # fail

q = 0.5
dim = 0
is_invariant_valid = invariant_check(input_tensor, dim, q)

if is_invariant_valid:
    print("Invariant check passed.")
else:
    print("Invariant check failed.")
