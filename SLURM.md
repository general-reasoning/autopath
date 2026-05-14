
    # cluster info
    ```
        # basis capabilities:
        scontrol show nodes
        # current state
        sinfo -N -l
        # jobs
        squeue
    ```
    # start an interactive job:
    ```
        # alloc: GPU
        #salloc --nodelist rhubarb -N 1 -n 1 --mem=8G --tres-per-node=gres/gpu:1
        salloc --nodelist rhubarb --nodes 1 --ntasks 1 --cpus-per-task 1 --mem=8G --gpus=1
        salloc --nodelist radish --nodes 1 --ntasks 1 --cpus-per-task 16 --mem=128G --gpus=1
        # alloc: CPU
        salloc --nodelist rhubarb -N 1 -n 1 --mem=2G --gpus=0
        salloc --nodelist radish -N 1 -n 1 --mem=2G --gpus=0
        # job params: OPTIONAL
        env | grep SLURM_
        env | grep SLURM_JOBID
        # start interactive shell:
        srun --jobid=$SLURM_JOBID --pty bash

        # on rhubarb: BEGIN
        nvidia-smi
        htop
        conda activate pancan-gigapath
        # ...
        exit
        # on rhubarb: END

        scancel $SLURM_JOBID

    # GPU
    export CUDA_VISIBLE_DEVICES="3"
