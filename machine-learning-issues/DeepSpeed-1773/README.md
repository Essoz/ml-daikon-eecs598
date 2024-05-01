# DeepSpeed Gradient overflow with fp16 enabled

## Environmental Setup

1. First install latest DeepSpeed version (version 0.13.3 2/18/2024) in the editable fashion.

```bash
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
pip3 install -r requirements/requirements.txt
pip3 install -e .
```

Note: If you still face dependency issues, you can use the install.sh script to install and check for the required dependencies.


2. Download bigscience/Megatron-DeepSpeed.

```bash
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed.git
```

Install the following dependencies by pip install -r requirements.txt. Do comment out the deepspeed line in the requirements.txt file before you proceed.

3. Install CUDA nvcc compiler (this is required for apex).

Please check the CUDA version shipped with your PyTorch installation.
Download and the install the CUDA nvcc compiler with the same CUDA version as PyTorch's. You can conveniently install it using the script here: https://github.com/TimDettmers/bitsandbytes/blob/main/install_cuda.sh

4. Install the apex library. Please go to the NVIDIA Apex repository and follow the installation instructions.

```bash
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

**NOTE**: For CUDA2.0 it is very likely to encounter compatibility issues. If so, please comment out the following lines before running the above code.

In setup.py:
```python
    # if (bare_metal_version != torch_binary_version):
    #     raise RuntimeError(
    #         "Cuda extensions are being compiled with a version of Cuda that does "
    #         "not match the version used to compile Pytorch binaries.  "
    #         "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
    #         + "In some cases, a minor-version mismatch will not cause later errors:  "
    #         "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
    #         "You can try commenting out this check (at your own risk)."
    #     )
```

## Run

1. Copy the following files inside the `Megatron-DeepSpeed` directory:

```
merges.txt
output.txt
run.sh
vocab.json
codeparrot_data.json (dataset, not contained in the repo)
```

2. Start `./run.sh` in new `screen` session:

```bash
screen -S mysession
./run.sh &> output.txt
```

To resume session, run:

```bash
screen -r mysession
```

The whole training log is saved in the file `output.txt`. In the experiment, the gradient overflow occurs at step `444` `14 hour 7 min` after the trainging process starts. The program throws an exception at step `456` `21 min` after the loss becomes `nan`, stopping the training process. 

```
raise Exception(
Exception: ExceptionCurrent loss scale already at minimum - cannot decrease scale anymore. Exiting run.: 
Current loss scale already at minimum - cannot decrease scale anymore. Exiting run.
```

## Root Causes

- FP16:

  - Sign bit: 1 bit
  - Exponent: 5 bits
  - Mantissa: 10 bits

- BF16:

  - Sign bit: 1 bit
  - Exponent: 8 bits
  - Mantissa: 7 bits

The main reason FP16 is more prone to gradient overflow compared to BF16 is due to the difference in exponent size. In FP16, the exponent is only 5 bits, which means it has a smaller range of representable exponents compared to BF16, which has an 8-bit exponent.

## Potential Ways to Detect the Bugs Automatically

TBD


