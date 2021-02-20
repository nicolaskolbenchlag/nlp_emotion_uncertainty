#!/bin/bash

#SBATCH --mem=20000
#SBATCH -J uncertainty

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty

source $START_HERE/venv/bin/activate
# python3 $START_HERE/MuSe-LSTM-Attention-baseline-model/extract_features/emotion_recognition/main.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --rnn_dr .3 --attn_dr .3 --out_dr .3 --uncertainty_approach monte_carlo_dropout --predict
python3 MuSe-LSTM-Attention-baseline-model/extract_features/emotion_recognition/main.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 1 --seed 314 --predict --attn --rnn_bi --loss cwCCC --uncertainty_approach cw_ccc