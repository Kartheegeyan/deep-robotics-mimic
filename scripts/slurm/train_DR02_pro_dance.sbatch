#!/bin/bash
# DR02_pro tracking on dataset/gmr/dance1_subject1.npz. A thin wrapper around
# train_DR02_pro.sbatch; same knobs, extra args go to train.py:
#
#   sbatch scripts/slurm/train_DR02_pro_dance.sbatch [extra train.py args...]
#
# The #SBATCH lines must stay in sync with train_DR02_pro.sbatch, since sbatch
# only reads directives from the submitted file.

#SBATCH --job-name=dr02_dance
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=raus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00

REPO=${REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null)}
: "${REPO:?submit from inside the deep-robotics-mimic checkout, or set REPO}"
export REPO
exec bash "$REPO/scripts/slurm/train_DR02_pro.sbatch" dataset/gmr/dance1_subject1.npz "$@"
