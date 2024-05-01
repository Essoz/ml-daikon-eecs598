# PYTORCH Issue 82206

## Environment Setup

Install the OLD pytorch version https://github.com/pytorch/pytorch/commit/ecd2c71871f8bf9a9fa4a4d875609b0922061a6f (Bug has been fixed in latest version)

## Run

`python main.py`

## What is the bug

When CPUOffload is enabled, the ShardedGradScaler.step seems to take forever to run.

## Possible Cause

When executing a ShardedGradScaler step in the context of cpu_offload, the function _foreach_non_finite_check_and_unscale_cpu_ is grindingly slow. This issue is due to the elementwise op dispatching/redispatching/execution that is engendered by the current approach to gradient tensor validation:
`pytorch/torch/distributed/fsdp/sharded_grad_scaler.py`

Lines 159 to 163 in `ecd2c71`
```python
 expected_device = grads[0].device 
 for grad in grads: 
     for tensor in grad: 
         if tensor.device != expected_device: 
             log.error( 
```

The subsequent isinf and isnan checks with associated any checks result in unscalable elementwise op dispatches:
`pytorch/torch/distributed/fsdp/sharded_grad_scaler.py`

Lines 173 to 181 in `ecd2c71`

```python
 if ( 
     torch.isinf(tensor).any().item() is True 
     or torch.isnan(tensor).any().item() is True 
 ): 
     found_inf.data = torch.tensor([1.0]) 
     break 
 else: 
     tensor.data *= inv_scale.item() 
```

This inefficency is of course hidden in the current FSDP tests given their (appropriately) trivial parameter dimensionality. In the perf analysis below, the example test configures only the final Linear(4, 8) module parameters to require grad, so there are 40 elements to iterate through. However, if one increases the dimensionality to a still-modest 320008 elements (changing the final module to Linear(40000,8)), the execution time/cpu cost of the test is dominated by the elementwise op dispatching/redispatching/execution of the any validation ops in this function.

## How to Fix

See https://github.com/pytorch/pytorch/pull/100108/files

Delete the elementwise op dispatching/redispatching/execution that is engendered by the current approach to gradient tensor validation.

## Potential Ways to Detect the Bug Automatically

1. `func_call_count` monitor: Count the number of dispatches for elementwise operations on tensors. This can involve monitoring the invocation of functions like torch.isinf, torch.isnan, and any subsequent any or item calls on tensors. By setting thresholds for expected counts in the context of specific operations, deviations that indicate inefficient handling (such as an excessive number of calls for relatively simple operations) can trigger alerts or flags for further investigation.

2. `Tensor Operation Counting Hooks`: Implement hooks or decorators in the development environment that automatically count and report the number of tensor operations performed during specific tasks or tests. By analyzing these counts, developers can identify and investigate operations that are performed more frequently than expected.