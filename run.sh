#!/bin/bash

#SBATCH --mem=10000
#SBATCH -J uncertainty

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty

source $START_HERE/venv/bin/activate
python3 $START_HERE/MuSe-LSTM-Attention-baseline-model/extract_features/emotion_recognition/main.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --predict --n_seeds 2 --seed 316 --attn --rnn_bi --loss tilted --uncertainty_approach quantile_regression