# env 
conda create -n autopath
conda activate autopath


# dbx
cd ~/dbx
pip install -e .


# dinov2
cd ~/dinov2
# set dependencies in ~/dinov2/requirements.txt as follows (to comport with what's below; 
# removing the troublesome cuml-cu11):
    --extra-index-url https://download.pytorch.org/whl/cu124
    torch==2.6.0+cu124
    torchvision==0.21.0+cu124
    omegaconf
    torchmetrics==1.8.0
    fvcore
    iopath
    xformers==0.0.29.post3
    submitit
    --extra-index-url https://pypi.nvidia.com
    ##cuml-cu11
# remove conda.yaml and conda-extras.yaml
pip install -e .

    # dependencies (in addition to or overriding those in the env, slideflow and autopath)
    # TODO: incorporate into the appropriate requirements file
    # N.B.: some of these packages may cause conflicts with those already in the env.
    #   In part this is why they are being installed later
    pip install jupyterlab==4.4.3
    pip install timm==1.0.15
    pip install xformers==0.0.20
    pip install dinov2 --extra-index-url https://pypi.nvidia.com # import dinov2 fails
    pip install omegaconf==2.3.0
    pip install fvcore==0.1.5.post20221221

    # for ImageNet
    pip install --upgrade torchvision=0.20.0 # implies torch==2.5.0, incompatible with xformers==0.0.20, fastai
    ### PROBLEM with xformers:
    ```
    NotImplementedError: No operator found for `memory_efficient_attention_forward` with inputs:
        query       : shape=(128, 197, 24, 64) (torch.float32)
        key         : shape=(128, 197, 24, 64) (torch.float32)
        value       : shape=(128, 197, 24, 64) (torch.float32)
        attn_bias   : <class 'NoneType'>
        p           : 0.0
    `flshattF` is not supported because:
        xFormers wasn't build with CUDA support
        dtype=torch.float32 (supported: {torch.bfloat16, torch.float16})
        Operator wasn't built - see `python -m xformers.info` for more info
    `tritonflashattF` is not supported because:
        xFormers wasn't build with CUDA support
        dtype=torch.float32 (supported: {torch.bfloat16, torch.float16})
        Only work on pre-MLIR triton for now
    `cutlassF` is not supported because:
        xFormers wasn't build with CUDA support
        Operator wasn't built - see `python -m xformers.info` for more info
    `smallkF` is not supported because:
        xFormers wasn't build with CUDA support
        max(query.shape[-1] != value.shape[-1]) > 32
        Operator wasn't built - see `python -m xformers.info` for more info
        unsupported embed per head: 64
    ```
    ### Potential FIX (Gemini):
    Reinstall xformers with CUDA-enabled PyTorch:
    Ensure you have the correct PyTorch version installed that's compatible with your desired CUDA version (e.g., CUDA 11.8 or 12.1), according to Stack Overflow.
    Uninstall your existing xformers installation: pip uninstall xformers.
    Install xformers using the appropriate PyTorch CUDA wheel: pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118 (for CUDA 11.8) or pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124 (for CUDA 12.4), according to Reddit.
    Verify the installation by checking if torch.cuda.is_available() returns True
    ### IMPL:
    * check cuda version
    import torch
    torch.version.cuda
    '12.4'
    * uninstall xformers
    pip uninstall xformers
    * reinstall xformers
    pip install -U xformers --index-url https://download.pytorch.org/whl/cu124
    #### this will reinstall torch to 2.5.0, among other things, 
    #### and seems to mess with torchvision, so
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124


# autopath
cd ~/autopath
pip install -e .


# tensorboard
conda activate autopath
tmux # tensorboard
cd $HOME/autopath/tensorboard
export TENSORBOARD_LOGDIR=$HOME/autopath/tensorboard
nohup tensorboard --logdir=$TENSORBOARD_LOGDIR --port 7007 > tensorboard.out &
# from a remote terminal
ssh -L 7007:127.0.0.1:7007 wheelbarrow

# detectron2
    #pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html # Example for CUDA 11.8 and PyTorch 2.1
    # Therefore, taking into account torch==2.6.0+cu124
    # pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu124/torch2.6/index.html # Example for CUDA 12.4 and PyTorch 2.6
    # the above fails, so installing from source:
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
    # Verify:
    python -m detectron2.utils.collect_env