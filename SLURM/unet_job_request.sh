#!/bin/bash -l
#SBATCH -J UnetTestJob
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=martuccifrance@gmail.com
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-02:00:00
#SBATCH --gpus=1
#SBATCH -p students-dev
#SBATCH --qos=normal

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"

srun -Q --immediate=10 --partition=students-dev --gres=gpu:1 --time 02:00:00 --pty bash $HOME/cv_p_13/unet_train.sh
