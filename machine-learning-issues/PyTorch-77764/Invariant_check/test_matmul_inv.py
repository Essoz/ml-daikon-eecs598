import torch

def invariant_check(input_zeros, input_ones):
    # Perform batched matrix multiplication on MPS device
    result = input_zeros @ input_ones
    
    # Count the number of ones in the result tensor
    num_ones = torch.sum(result == 1).item()
    
    return num_ones

# Example usage:
input_zeros = torch.zeros(911, 9, 1, device=torch.device("mps"))
input_ones = torch.ones(1, 32769, device=torch.device("mps"))

# Run the invariant check multiple times
num_runs = 5  # Adjust the number of runs as needed
invariant_results = []

for _ in range(num_runs):
    num_ones = invariant_check(input_zeros, input_ones)
    invariant_results.append(num_ones)

# Check if the number of ones is consistent across runs
if all(result == invariant_results[0] for result in invariant_results):
    print("Invariant check passed.")
else:
    print("Invariant check failed.")
