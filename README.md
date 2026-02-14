cd ~

mkdir PNP

cd ./PNP

python3 -m venv .venv

source ./.venv/bin/activate

# part 1 :Install the open-source library **EasyFHE
git clone --recursive -b release-1.0 git@github.com:jizhuoran/EasyFHE.git

cd EasyFHE

pip install -r requirements.txt

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

export PATH=$PATH:/usr/local/cuda/bin

export CUDA_HOME=$CUDA_HOME:/usr/local/cuda

USE_DISTRIBUTED=0 USE_MKLDNN=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 USE_NINJA=OFF USE_ROCM=0 python3 setup.py develop --install-dir=~/torch/



# part 2:Extract the **anonymous GitHub** files to the specified folder
unzip ICML_2026_PackCNN.zip -d ./PNP/EasyFHE/ICML_2026_PackCNN


# part 3:Run
#Before running the main program, please run the setup script to configure the required files.
cd ./PNP/EasyFHE/ICML_2026_PackCNN
python script.py 

# NOTE: During the first-time generation, we need to preprocess (preload/encode) the weights,
# which takes a relatively long time (about 10 minutes)
# NOTE: The first-time generation will produce approximately 60GB of .pkl files.
# If there is not enough disk space, please change the DATA_DIR path to another directory with sufficient storage.
cd ../
python3 -m ICML_2026_PackCNN.PackCNN 0 0

# NOTE: After the first run, the .pkl files will be generated.
# At this point, the code sections responsible for generating the .pkl files should be ignored.
# Accordingly, the runtime command should be modified to: (replace the .pkl filename with the one generated during the first run)
python3 -m ICML_2026_PackCNN.PackCNN 1 /encode_20260128_150521.pkl
