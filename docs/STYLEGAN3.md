# stylegan3-slideflow:
## Clone https://github.com/jamesdolezal/stylegan3-slideflow
conda env create -f stylegan3-slideflow/environment.yml
conda activate stylegan3
export PYTHONPATH=$HOME/stylegan3-slideflow:$PYTHONPATH

# slideflow
pip install versioneer
pip install -e $HOME/slideflow

# dependencies
pip install torchvision ipython

# test
python -c "from stylegan3 import dnnlib, legacy, utils"

# RUN:
## bash
export CUDA_VISIBLE_DEVICES=1
## ipython
import torch
from slideflow.gan.stylegan3.stylegan3 import dnnlib, legacy, utils

with dnnlib.util.open_url('/mnt/labshare/MODELS/HistoXGAN/FINAL_MODELS/CTransPath/snapshot.pkl') as f:
   G = legacy.load_network_pkl(f)['G_ema'].to('cuda')

t = torch.rand(768) # 768 is the ctranspath embedding dimension
img = G(t[None].cuda(), 0, noise_mode='const')
#To convert to HWC format and 0-255 scale
img = (img + 1) * (255/2)
img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
