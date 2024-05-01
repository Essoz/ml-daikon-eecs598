# PYTORCH-77764 MPS Issues

https://github.com/pytorch/pytorch/issues/101878

https://github.com/pytorch/pytorch/issues/111634

## Environment Setup

Install the latest version of PyTorch will suffice (as of Feb 4, 2024).

## Run

Check in the issue folder
`pwd`
`*/PyTorch-77764/`
Run `pytest` **on a Mac OS Device that supports MPS**

## What is the bug

**Inconsistent and immature implementation of functions for MPS as compared to other devices.**

* incorrect return value for `torch.quantile` on mps device
* incorrect return gradients by the backward method on mps device
* incorrect return for matmul operation on mps device (should be a all-zero matrix but element '1' occurs)

## Possible Cause

* `t.transpose()` and `t.unsqueeze()` functions inside `torch.quantile` provide a strided view that the sort() MPS implementation doesn't handle well
* MPS framework related

## How to Fix

TBD

## Potential Ways to Detect the Bug Automatically

1. Testing: create unit tests/logging/monitoring that specifically target the behavior of `torch.quantile` when running on MPS devices
2. Invariant check:
    1. Consistency Across Platforms:
       Different devices (e.g. CPU, CUDA) should produce the same result for the same operation (with acceptable errors). We can compare the MPS result with the same operation on other devices (e.g. CPU, CUDA) to check if the results are consistent.
    2. Determinism Across Runs:
       The same operation should produce the same result across runs. We can capture the result of the same operation across runs and compare them to check if the results are consistent.
       **Note**

### Example Code for Invariant Check

1. [`test_quantile_inv.py`](./Invariant_check/test_quantile_inv.py):
Consistency Across Platforms: the invariant check compares the MPS result with the reference result using torch.allclose to check if they are close within a specified tolerance.
2. [`test_matmul_inv.py`](./Invariant_check/test_matmul_inv.py):
Determinism Across Runs: the invariant check considers the bug if the number of ones in the result tensor varies across runs.  
