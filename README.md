# PackCNN

> **GPU runtime note**
>
> PackCNN supports PPML inference based on EasyFHE/GPU-FHE. Because CPU-based execution is very slow for this workload, this README uses the GPU version as the default example.
>
> The GPU used for running this example should have **at least 80GB of GPU memory**, such as an **NVIDIA H100**.

## Project Structure

```text
PackCNN/
├── script.py
├── run.py
├── README.md
├── data/
│   ├── cifar10_resnet20-4118986f.pt
│   ├── params1.npz
│   ├── params2.npz
│   └── test_batch.bin
└── pack/
    ├── bsgs.py
    ├── config.py
    ├── conv.py
    ├── crypto.py
    ├── data.py
    ├── encoding.py
    ├── model.py
    ├── pipeline.py
    └── utils.py
```

## Module Overview

- `script.py`: Setup script adapted for the EasyFHE runtime environment.
- `run.py`: Lightweight command-line entry point.
- `pack/pipeline.py`: Main inference pipeline and execution orchestration.
- `pack/conv.py`: Packed homomorphic convolution, boundary handling, and downsampling logic.
- `pack/bsgs.py`: BSGS-style plaintext weight preparation.
- `pack/crypto.py`: Encryption utilities, homomorphic ReLU/Aespa operations, and bootstrapping helpers.
- `pack/encoding.py`: Ciphertext checkpointing and pre-encoded weight loading.
- `pack/data.py`: CIFAR batch loading and input packing.
- `pack/model.py`: Model weight extraction, average pooling, fully connected layer execution, and plan selection.
- `pack/config.py`: Runtime path configuration and global configuration state.
- `pack/utils.py`: Shared mathematical and indexing utilities.

## Installation

Install EasyFHE first:

```bash
cd ~
mkdir -p PNP
cd ./PNP

python3 -m venv .venv
source ./.venv/bin/activate
# Clone EasyFHE using ONE of the following methods.

# Option 1: SSH, recommended if your GitHub SSH key is configured.
git clone --recursive -b release-1.0 git@github.com:jizhuoran/EasyFHE.git

# Option 2: HTTPS, use this if SSH is not available.
# Uncomment the following command and comment out the SSH command above if needed.
# git clone --recursive -b release-1.0 https://github.com/jizhuoran/EasyFHE.git

cd EasyFHE
pip install -r requirements.txt

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda

USE_DISTRIBUTED=0 USE_MKLDNN=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 USE_NINJA=OFF USE_ROCM=0 python3 setup.py develop --install-dir=~/torch/
```

If the build fails with a `Cannot find CUB` error, clean the previous build:

```bash
python3 setup.py clean
```

Configure the CUDA/CUB paths explicitly and rebuild:

```bash
export CUB_INCLUDE_DIR=/usr/local/cuda-12.8/include/cub
export CMAKE_INCLUDE_PATH=/usr/local/cuda-12.8/include/cub:$CMAKE_INCLUDE_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

USE_DISTRIBUTED=0 USE_MKLDNN=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 USE_NINJA=OFF USE_ROCM=0 python3 setup.py develop --install-dir=~/torch/
```

After EasyFHE is installed, extract or place this project under the EasyFHE/GPU-FHE workspace:

```bash
unzip PackCNN.zip -d ~/PNP/EasyFHE/PackCNN
```

## Runtime Data Directory

The runtime data directory is configured in `pack/config.py`:

```python
os.environ["DATA_DIR"] = os.path.join(project_root, "PackCNN", "data")
```

This directory is used to store GPU-FHE context files, encrypted input checkpoints, and pre-encoded weight `.pkl` files.

Make sure the directory has sufficient free disk space before running the first-time preprocessing stage.

## Run

Run all commands from the project directory:

```bash
cd ~/PNP/EasyFHE/PackCNN
```

Before running the main program, execute the setup script to configure the required files:

```bash
python3 script.py
```

### First-Time Preprocessing and Generation

Run the following command to perform first-time preprocessing and generate the encoded weight file:

```bash
python3 run.py 0 0
```

Notes:

- First-time preprocessing performs weight encoding and preloading, which may take a long time.
- The generated `.pkl` files may require approximately 60 GB of storage.
- Ensure that `DATA_DIR` has enough available disk space before starting preprocessing.

### Inference with Generated Encoded Weights

After the encoded `.pkl` file has been generated, run inference by passing the generated filename:

```bash
python3 run.py 1 /encode_20260503_143126.pkl
```

## Entry Points

`run.py` contains only the command-line entry logic.

All core execution logic is implemented in `pack/pipeline.py`, with supporting functionality provided by the modules under `pack/`.
