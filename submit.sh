#!/bin/bash
#
#SBATCH --job-name=test_nln
#
# Account:
#SBATCH --account=co_esmath
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# Quality of Service:
#SBATCH --qos=esmath_gpu3_normal
#
# Number of nodes: 
#SBATCH --nodes=1
#
# Processors per task 
#SBATCH --cpus-per-task=8
#
#SBATCH --gres=gpu:GTX2080TI:4
#
#SBATCH --time=00:01:00
#



module load python
source ~/jaxenv/bin/activate

# load gpu related 
module load gcc openmpi
module load cuda/11.2
module load cudnn/7.0.5
export CUDA_PATH=/global/software/sl-7.x86_64/modules/langs/cuda/11.2
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH


XLA_FLAGS=--xla_gpu_cuda_data_dir=/global/software/sl-7.x86_64/modules/langs/cuda/11.2 python ~/cancellations/run.py

#module load gnu-parallel/2019.03.22 
#parallel -j $SLURM_CPUS_ON_NODE XLA_FLAGS=--xla_gpu_cuda_data_dir=/global/software/sl-7.x86_64/modules/langs/cuda/11.2 python runtest.py 20000 30 ::: `seq 10`
