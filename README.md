# PackCNN

PackCNN is a PPML inference implementation built on EasyFHE/GPU-FHE. 

## Project Structure

```text
PackCNN/
├── script.py
├── run.py
├── README.md
├── data/
│   ├── cifar10_resnet20-4118986f.pt
│   ├── params1.npz
│   └── params2.npz
│   └── test_batch.bin
└── pack/
    ├── __init__.py
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

Module responsibilities:
- `script.py`: script adapted for EasyFHE 
- `run.py`: short command-line entry point.
- `pack/pipeline.py`: main inference orchestration.
- `pack/conv.py`: packed homomorphic convolution, edge handling, and downsampling.
- `pack/bsgs.py`: BSGS-style plaintext weight preparation.
- `pack/crypto.py`: encryption helpers, homomorphic ReLU/Aespa, and bootstrapping.
- `pack/encoding.py`: ciphertext checkpointing and pre-encoded weight loading.
- `pack/data.py`: CIFAR batch loading and input packing.
- `pack/model.py`: model weight extraction, average pooling, fully connected layer, and plan selection.
- `pack/config.py`: runtime paths and configuration state.
- `pack/utils.py`: shared math/index utilities.

## Installation

Install EasyFHE first:

```bash
cd ~
mkdir -p PNP
cd ./PNP

python3 -m venv .venv
source ./.venv/bin/activate

git clone --recursive -b release-1.0 git@github.com:jizhuoran/EasyFHE.git
cd EasyFHE

pip install -r requirements.txt

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda

USE_DISTRIBUTED=0 USE_MKLDNN=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 USE_NINJA=OFF USE_ROCM=0 python3 setup.py develop --install-dir=~/torch/
```

Extract or place this project under the EasyFHE/GPU-FHE workspace:

```bash
unzip PackCNN.zip -d ./PNP/EasyFHE/PackCNN
```

## Runtime Data Directory

The code keeps data setting in `pack/config.py`:

```python
os.environ["DATA_DIR"] = "/data/test/data"
```

This directory is used for GPU-FHE context files, encrypted input checkpoints, and encoded weight `.pkl` files. Make sure it has enough free space before first-time preprocessing.

## Run
Run commands from the project directory:

```bash
cd ./PNP/EasyFHE/PackCNN
```

Before running the main program, please run the setup script to configure the required files:

```bash
python script.py
```

First-time preprocessing/generation:

```bash
python run.py 0 0
```

Notes:

- First-time preprocessing encodes/preloads weights and can take a long time.
- The generated `.pkl` files can require about 60GB of storage.
- Make sure `DATA_DIR` has enough free space before running.

After the encoded `.pkl` file has been generated, run inference by passing the generated filename:

```bash
python run.py 1 /encode_20260128_150521.pkl
```



## Entry Points

`run.py` contains only this entry logic. All business logic is in `pack/pipeline.py` and the supporting modules under `pack/`.
