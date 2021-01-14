### Experiments on uncertainty measurement in NLP

Baseline model for MuSe-CaR database adjusted for approaches to decrease uncertainty `MuSe-LSTM-Attention-baseline-model/extract_features/emotion_recognition/main.py`:
* Quantille Regression (`--uncertainty_approach quantile_regression`, tilted loss needs to be specified seperately)
* Monte Carlo Dropout (`--uncertainty_approach monte_carlo_dropout`, dropout rates need to be specified additionally)
* Correlation-weighted loss/CCC (not supported per parameterization)