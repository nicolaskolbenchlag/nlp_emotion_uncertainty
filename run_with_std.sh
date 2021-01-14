#!/bin/bash

#SBATCH --mem=30000
#SBATCH -J cwCCC

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty

source $START_HERE/venv/bin/activate
python3 $START_HERE/MuSe-LSTM-Attention-baseline-model/extract_features/emotion_recognition/main_stdv.py --feature_set fasttext --emo_dim_set valence --epochs 100 --refresh --predict --n_seeds 10 --seed 314 --attn --rnn_bi --loss cccStd