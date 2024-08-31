#!/bin/bash
#SBATCH -J ESMCPU
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH -p A30
#SBATCH -w gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --export=ALL
#SBATCH -o log/output.log
#SBATCH -e log/error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=925201392@qq.com

conda activate pyG
mkdir -p log
#parallel -j 8 < ./run.sh
python ~/embedding/ESM/esm/scripts/extract.py  esm2_t36_3B_UR50D  wild.fa output --repr_layers -1 --include mean per_tok --truncation_seq_length 2000
