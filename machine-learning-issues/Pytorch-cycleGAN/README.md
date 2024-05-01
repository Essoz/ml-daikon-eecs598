# PYTORCH-CycleGAN Issue 619



## Environment Setup

Install the latest version of PyTorch will suffice (as of Feb 4, 2024).

## Run


- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install related dependancies
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.
  - For Repl users, please click [![Run on Repl.it](https://repl.it/badge/github/junyanz/pytorch-CycleGAN-and-pix2pix)](https://repl.it/github/junyanz/pytorch-CycleGAN-and-pix2pix).

### CycleGAN train/test
- Download a CycleGAN dataset (e.g. maps):
```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```


## What is the bug

** Training Stops Randomly around epoch 70+.**

> The training process stops, without giving any error or warning.
  (my terminal just states: end of epoch 71 ..., and then nothing happens anymore)
  If I check the state of my graphics-card (using nvidia-smi), I can see that the GPU memory is still allocated by python, but the GPU-usage is 0% (normally during training, this goes between about 100 and 95%).

> If I then stop the script and restart training with the --continue_train option, it does work and finishes up after 200 epochs.

## Possible Cause

* Direct Cause:

The direct cause is the dead loop in the pytorch [dataloader.py](https://github.com/pytorch/pytorch/blob/a24163a95edb193ff7b06e98cd69bf7cfd4c0d2f/torch/utils/data/dataloader.py#L94-L111) where the watchdog is always alive and always continues

```python
    while True:
        try:
            r = index_queue.get(timeout=MANAGER_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            if watchdog.is_alive():
                continue
            else:
                break
        if r is None:
            break
        idx, batch_indices = r
        try:
            samples = collate_fn([dataset[i] for i in batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))
            del samples
```

* Root Cause:

Probably related to the connection issue caused by `visdom`

## How to Fix

Verified Solution: set `display_id=0` option which disabled `visdom`

## Potential Ways to Detect the Bug Automatically

1. Runtime resource usage monitor: (universal to all deadlock/infinite loop related issues)


2. Data-driven diagnostics (loss, model weight, etc): not applicable in these scenario since loss and model weight are normal.

- In `visualization.ipynb`

![image](https://github.com/OrderLab/machine-learning-issues/assets/97345341/a22beb0c-708a-44a7-b7a4-2f54ac033b72)


![image](https://github.com/OrderLab/machine-learning-issues/assets/97345341/55b66974-bb2c-4809-8741-00ea6a5fe49b)





