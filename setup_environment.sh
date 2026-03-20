conda_path=""

## create conda environment
source ~/.bashrc;
if [[ $conda_path == "" ]]; then
    source ~/miniconda3/etc/profile.d/conda.sh
else
    source $conda_path/miniconda3/etc/profile.d/conda.sh
fi
conda init

# create templateMatching SAM environment
conda create -n templateMatchingSAM python=3.12
conda activate templateMatchingSAM
conda install pip==25.2
# 
# # create folder if not existent
mkdir InpaintAnything
cd InpaintAnything

# clone Inpaint Anything repo
git clone https://github.com/geekyutao/Inpaint-Anything .
git checkout 5bfa9f3d1a829d71bddecf28c228952dbf825e3a


# replace remove_anything file and __init__ file
cd ..
scp InpaintAnything_replacements/remove_anything.py InpaintAnything/remove_anything.py
scp InpaintAnything_replacements/__init__.py 
cd InpaintAnything

# install packages
python3 -m pip install numpy==1.26.1 -r lama/requirements.txt 
python3 -m pip install numpy==1.26.1 -e segment_anything --config-settings editable_mode=compat
python3 -mpip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install numpy==1.26.1 hydra-core==1.3.2 sam2==1.1.0 easyocr supervision lpips clip scikit-learn

# copy checkpoints
mkdir ./pretrained_models/
wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1ZfKal2o7zZUfjCFYGmKz9w1BLwU4TYX7&confirm=t" -O ./pretrained_models/sam_vit_h_4b8939.pth

mkdir ./pretrained_models/big-lama/
mkdir ./pretrained_models/big-lama/models/
wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1d5M-c5Ij8kMGx6aeCmnTKQhf5zWYAyi0&confirm=t" -O ./pretrained_models/big-lama/models/best.ckpt
wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1F6aP4DI_FjVXG9CSBTa-Avp7CMPJM8k0&confirm=t" -O ./pretrained_models/big-lama/config.yaml

# install sam3
cd ..
mkdir sam3/
cd sam3/
git clone https://github.com/facebookresearch/sam3.git .
git checkout f6e51f59500a87c576c2df2323ce56b9fd7a12de
python3 -m pip install -e .
python3 -m pip install einops decord psutil pycocotools setuptools==80.10.2

# load sam2 checkpoint
cd ..
mkdir ./sam-2_checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O ./sam-2_checkpoints/sam2.1_hiera_large.pt

# sam3 checkpoint has to be downloaded manually as you have to apply for the checkpoint
mkdir ./sam-3_checkpoints
