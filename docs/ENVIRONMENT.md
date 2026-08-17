# Environment Setup

This guide creates a fresh environment for Uni-Sign without using the old pinned `requirements.txt` directly.

The original installation instructions install `requirements.txt` as-is. That file pins an older Python 3.9-era stack, includes TensorFlow even though it is not needed for the current training path, and hardcodes CUDA-specific PyTorch wheels that may not match current NVIDIA, AMD ROCm, CPU-only, or cluster environments. The commands below install the same project-critical packages in a hardware-specific order, with PyTorch selected explicitly for the target system.

The tested baseline is:

- Python 3.12
- PyTorch 2.12.0
- Transformers 4.44.2
- DeepSpeed 0.17.4
- No TensorFlow

Use Linux or WSL2 for training with DeepSpeed. Native Windows can work for local imports and debugging, but DeepSpeed custom CUDA ops are much easier to handle on Linux.

## 1. Create Environment

```bash
conda create -y -n uni-sign-slt python=3.12 pip
conda activate uni-sign-slt
python -m pip install --upgrade pip
```

## 2. Install PyTorch

Choose exactly one command for your machine.

### NVIDIA CUDA 13.0

```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
```

### NVIDIA CUDA 13.2

```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu132
```

### NVIDIA CUDA 12.6

```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
```

### AMD ROCm 7.2

Linux only:

```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

### CPU Only

```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
```

## 3. Install Project Dependencies

```bash
pip install accelerate==1.0.1 decord==0.6.0 einops==0.8.1 matplotlib==3.9.4 opencv-python-headless==4.12.0.88 pandas==2.2.3 rich==13.9.4 rouge==1.0.1 rouge-score==0.1.2 sacrebleu==2.4.3 scikit-image==0.25.2 scikit-learn==1.8.0 scipy==1.17.1 seaborn==0.13.2 tensorboard==2.18.0 timm==0.9.16 tokenizers==0.19.1 tqdm==4.67.3 transformers==4.44.2 sentencepiece==0.2.0 wandb==0.28.0 portalocker==2.10.1 datasets==4.0.0 evaluate==0.4.3
```

## 4. Install DeepSpeed

### Linux or WSL2

For a simple install without precompiled DeepSpeed ops:

```bash
DS_BUILD_OPS=0 pip install deepspeed==0.17.4
```

For a CUDA toolkit machine where you want DeepSpeed to compile compatible ops during install:

```bash
DS_BUILD_OPS=1 pip install deepspeed==0.17.4
```

Check the install:

```bash
python -m deepspeed.env_report
```

### Native Windows

Use WSL2 if you need reliable DeepSpeed training. For native Windows, first try:

```powershell
$env:DS_BUILD_OPS = "0"
pip install --no-build-isolation deepspeed==0.17.4
```

If this fails with `CUDA_HOME does not exist` during metadata generation, install DeepSpeed on Linux/WSL2 or build a patched local wheel with disabled op compatibility probes.

## 5. Optional Demo Dependencies

For pose extraction and online inference:

```bash
pip install --no-deps -e demo/rtmlib-main
pip install onnxruntime-gpu
```

For CPU-only ONNX Runtime:

```bash
pip install --no-deps -e demo/rtmlib-main
pip install onnxruntime
```

## 6. Validate

Run from the repository root:

```bash
python -c "import torch, deepspeed; import models, utils, datasets, SLRT_metrics; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('deepspeed', deepspeed.__version__); print('project imports ok')"
pip check
```

For the demo package:

```bash
python -c "import onnxruntime, rtmlib; print('onnxruntime', onnxruntime.__version__, onnxruntime.get_available_providers()); print('rtmlib ok')"
```
