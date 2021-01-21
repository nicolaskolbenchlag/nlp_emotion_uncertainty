### Experiments on uncertainty measurement in NLP

Baseline model for MuSe-CaR database adjusted for approaches to decrease uncertainty `MuSe-LSTM-Attention-baseline-model/extract_features/emotion_recognition/main.py`:
* Quantille Regression (`--uncertainty_approach quantile_regression`, tilted loss needs to be specified seperately)
* Monte Carlo Dropout (`--uncertainty_approach monte_carlo_dropout`, dropout rates need to be specified additionally)
* Correlation-weighted loss/CCC

Running:

* For running Quantile Regression: `$ python3 MuSe-LSTM-Attention-baseline-model/extract_features/emotion_recognition/main.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 10 --seed 314 --predict --attn --rnn_bi --loss tilted --uncertainty_approach quantile_regression`

* For running Monte Carlo Dropout: `$ python3 MuSe-LSTM-Attention-baseline-model/extract_features/emotion_recognition/main.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --predict --n_seeds 10 --seed 314 --attn --rnn_bi --loss ccc --rnn_dr .3 --attn_dr .3 --out_dr .3`

* For running cwCCC: `$ python3 MuSe-LSTM-Attention-baseline-model/extract_features/emotion_recognition/main.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 10 --seed 314 --predict --attn --rnn_bi --loss cwCCC --uncertainty_approach cw_ccc`