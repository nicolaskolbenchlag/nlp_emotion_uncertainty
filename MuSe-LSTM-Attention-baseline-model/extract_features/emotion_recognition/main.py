# *_*coding:utf-8 *_*
import argparse
import os
import sys
from datetime import datetime
from dateutil import tz
import numpy as np
import torch

from train import train_model, evaluate, predict, evaluate_mc_dropout
from model import Model
from dataset import MyDataset
import utils
import config

import train

import correlation_among_annotators

# parse parameters
def parse_params():
    parser = argparse.ArgumentParser(description='Multimodal emotion recognition')

    # Data
    parser.add_argument('--feature_set', nargs='+', required=True, help='feature set')
    parser.add_argument('--emo_dim_set', nargs='+', default=['arousal'],
                        help='emotion dimensions set ("arousal", "valence")')
    parser.add_argument('--segment_type', type=str, default='normal', choices=['normal', 'id'],
                        help='how to segment video samples?')
    parser.add_argument('--normalize', action='store_true',
                        help='normalize features?')
    parser.add_argument('--norm_opts', type=str, nargs='+', default=['n'],
                        help='normalize option for each feature ("y": normalize, "n": do not normalize)')
    parser.add_argument('--win_len', type=int, default=200,
                        help='window length to segment features (default: 200 frames)')
    parser.add_argument('--hop_len', type=int, default=100,
                        help='hop length to segment features (default: 100 frames)')
    parser.add_argument('--add_seg_id', action='store_true',
                        help='whether add segment id into the feature (default: False)')
    parser.add_argument('--annotator', type=int, default=None,  # [2, 4, 7, 5, 8]
                        help='Learn on raw annotator targets. (default: None)')
    # Architecture
    parser.add_argument('--d_rnn', type=int, default=64,
                        help='number of hidden states in rnn (default: 64)')
    parser.add_argument('--rnn_n_layers', type=int, default=1,
                        help='number of layers for rnn (default: 1)')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of layers for self attention (default: 1)')
    parser.add_argument('--attn', action='store_true',
                        help='whether add self_attention in the model. (default: False)')
    parser.add_argument('--transformer', action='store_true',
                        help='whether to add the MMT to the model. [not currently working] (default: False)')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='number of heads for self attention (default: 4)')
    parser.add_argument('--rnn', type=str, default='lstm', choices=['lstm', 'gru'],
                        help='type of rnn (default: lstm)')
    parser.add_argument('--rnn_bi', action='store_true',
                        help='whether rnn is bidirectional or not')
    parser.add_argument('--d_out_fc', type=int, default=64,
                        help='dimension of output layer (default: 64)')
    parser.add_argument('--out_biases', type=float, nargs='+',
                        help='biases of output layer')
    parser.add_argument('--loss', type=str, default='ccc', choices=['ccc', 'mse', 'l1', "tilted", "tilted_dyn", "tiltedCCC", "cwCCC"],
                        help='loss function (default: ccc)')
    parser.add_argument('--loss_weights', nargs='+', type=float,
                        help='loss weights for total loss calculation')
    parser.add_argument('--label_smooth', type=int, default=None,
                        help='the kernel size of avgpool1d for smoothing label in the training')
    parser.add_argument('--label_preproc', type=str, default='None', choices=['standard', 'norm', 'None', 'savgol'],
                        help='type of label preprocessing; choices: None, standard, norm, savgol (default: None)')
    parser.add_argument('--savgol_window', type=int, default=51, help='window length if label_preproc is set to savgol')
    parser.add_argument('--savgol_polyorder', type=int, default=3, help='polyorder if label_preproc is set to savgol')

    # Dropouts
    parser.add_argument('--rnn_dr', type=float, default=0.0,
                        help='dropout rate for rnn')
    parser.add_argument('--attn_dr', type=float, default=0.0,
                        help='dropout rate for self-attention')
    parser.add_argument('--out_dr', type=float, default=0.0,
                        help='dropout rate for output layer')
    parser.add_argument('--l2_penalty', type=float, default=0.0,
                        help='l2 penalty for network parameter')

    # Tuning
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='batch size')
    parser.add_argument('--early_stop', type=int, default=15,
                        help='number of epochs to early stop training')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='initial learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='plateau', choices=['step', 'plateau'],
                        help='scheduler of learning rate (default: plateau)')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='when to decay learning rate')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='learning rate decaying factor')
    parser.add_argument('--min_lr', type=float, default=0.0,
                        help='min learning rate when using "ReduceLROnPlateau"')
    parser.add_argument('--clip', type=float, default=0,
                        help='gradient clip value')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=1,
                        help='frequency of result logging')
    parser.add_argument('--seed', type=int, default=314,
                        help='random seed')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='number of random seeds')
    parser.add_argument('--gpu', type=str,
                        help='gpu id')
    parser.add_argument('--cache', action='store_true',
                        help='whether cache data. (default: False)')
    parser.add_argument('--log', action='store_true',
                        help='whether log file. (default: False)')
    parser.add_argument('--log_extensive', action='store_true',
                        help='whether to log training progress. (default: False)')
    parser.add_argument('--view', action='store_true',
                        help='plot results from fusion (default: False)')

    parser.add_argument('--save', action='store_true',
                        help='whether save best model. (default: False)')
    parser.add_argument('--predict', action='store_true',
                        help='whether make predictions. (default: False)')
    parser.add_argument('--save_dir', type=str,
                        help='directory used to save model prediction')
    parser.add_argument('--refresh', action='store_true',
                        help='whether construct data from scratch. (default: False)')

    
    # NOTE: choose uncertainty approach
    parser.add_argument('--uncertainty_approach', type=str, choices=[None, 'quantile_regression', 'monte_carlo_dropout', 'cw_ccc'])

    # parse
    args = parser.parse_args()
    return args


def get_dataloaders(data, predict=False):
    sample_counts = []
    data_loader = {}
    for partition in data.keys():
        set_ = MyDataset(data, partition)
        sample_counts.append(len(set_))

        batch_size = params.batch_size if partition == 'train' and not predict else 1
        shuffle = True if partition == 'train' else False
        data_loader[partition] = torch.utils.data.DataLoader(set_, batch_size=batch_size, shuffle=shuffle,
                                                             num_workers=4)
    print(f'Samples in partitions: {*sample_counts,}')
    return data_loader


def main(params):
    # load data
    print('Constructing dataset and data loader ...')

    ########################################
    if params.uncertainty_approach == 'cw_ccc':
        data_loader, data_loader_gt, data = correlation_among_annotators.get_correlations_data_loader(params)
    
    else:
        data = utils.load_data(params, params.feature_set, params.emo_dim_set, params.normalize, params.label_preproc,
                            params.norm_opts, params.segment_type, params.win_len, params.hop_len, save=params.cache,
                            refresh=params.refresh, add_seg_id=params.add_seg_id, annotator=params.annotator)
        data_loader = get_dataloaders(data)
        if params.annotator is not None:
            data_gt = utils.load_data(params, params.feature_set, params.emo_dim_set, params.normalize, params.label_preproc,
                                    params.norm_opts, 'None', params.win_len, params.hop_len, save=params.cache,
                                    refresh=params.refresh, add_seg_id=params.add_seg_id, annotator=None)
            data_loader_gt = get_dataloaders(data_gt, True)
            print('-' * 50)
        else:
            data_loader_gt = None
    ########################################

    # check params
    if params.out_biases is None:
        params.out_biases = [0.0] * len(params.emo_dim_set)
    if params.loss_weights is None:
        if len(params.emo_dim_set) == 2:
            params.loss_weights = [0.5, 0.5]  # default: 0.5 * arousal + 0.5 * valence
        else:
            params.loss_weights = [1]
    assert len(params.emo_dim_set) == len(params.loss_weights) and len(params.loss_weights) == len(params.out_biases)
    assert (not params.add_seg_id and params.transformer) or not params.transformer
    assert (params.transformer and len(params.feature_set) == 3) or not params.transformer

    # additional params
    params.d_in = data_loader['train'].dataset.get_feature_dim()
    print(f'Input feature dim: {params.d_in}.')
    params.d_out = len(params.emo_dim_set)
    params.feature_dims = data['train']['feature_dims']

    # seed setting
    seeds = range(params.seed, params.seed + params.n_seeds)
    val_losses, val_cccs, val_pccs, val_rmses, best_model_files = [], [], [], [], []
    test_cccs, test_pccs, test_rmses = [], [], []

    for seed in seeds:
        params.current_seed = seed
        torch.manual_seed(seed)
        if params.gpu is not None and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # construct model
        model = Model(params)

        if params.log_extensive:
            print('*' * 100)
            print(f'Using seed "{seed}"')
            print('*' * 100)
            print(model)

        # train model
        print('=' * 50)
        print('Training model... [seed {}]'.format(params.current_seed))
        val_loss, val_ccc, val_pcc, val_rmse, best_model_file = \
            train_model(model, data_loader, params)
        
        ########################################
        if params.uncertainty_approach == None or params.uncertainty_approach == 'cw_ccc':
            test_ccc, test_pcc, test_rmse = evaluate(model, data_loader['test'], params)
        
        elif params.uncertainty_approach == "quantile_regression":
            test_ccc, test_pcc, test_rmse = train.evaluate_quantile_regression(model, data_loader['test'], params)
        
        elif params.uncertainty_approach == "monte_carlo_dropout":
            test_ccc, test_pcc, test_rmse = train.evaluate_mc_dropout(model, data_loader['test'], params)
        
        else:
            raise NotImplementedError()
        ########################################

        val_losses.append(val_loss)
        val_cccs.append(val_ccc)
        val_pccs.append(val_pcc)
        val_rmses.append(val_rmse)
        best_model_files.append(best_model_file)

        test_cccs.append(test_ccc)
        test_pccs.append(test_pcc)
        test_rmses.append(test_rmse)

        print('On Test: CCC {:7.4f} | PCC {:7.4f} | RMSE {:7.4f}'.format(test_ccc[0], test_pcc[0], test_rmse[0]))

        if params.current_seed == seeds[-1]:
            print('=' * 50)
        if params.log_extensive:
            print('*' * 100)
            print(f'Seed "{params.current_seed}" over!')
            print('*' * 100)

    mean_val_cccs = [np.mean(val_ccc) for val_ccc in val_cccs]
    best_idx = mean_val_cccs.index(max(mean_val_cccs))
    best_val_ccc, best_mean_val_ccc = val_cccs[best_idx], mean_val_cccs[best_idx]

    mean_test_cccs = [np.mean(test_ccc) for test_ccc in test_cccs]
    best_test_ccc, best_mean_test_ccc = test_cccs[best_idx], mean_test_cccs[best_idx]

    if params.annotator is not None:
        print(f'On annotator labels:\tBest\t[Val CCC] for seed "{seeds[best_idx]}":\t{best_mean_val_ccc:7.4f}')
        print(f'On annotator labels:\t\t[Test CCC] for seed "{seeds[best_idx]}":\t{best_mean_test_ccc:7.4f}')
        print('-' * 100)
    else:
        print(f'On ground-truth labels:\tBest\t[Val CCC] for seed "{seeds[best_idx]}":\t{best_mean_val_ccc:7.4f}')
        print(f'On ground-truth labels:\t\t[Test CCC] for seed "{seeds[best_idx]}":\t{best_mean_test_ccc:7.4f}')
        print('-' * 100)

    if data_loader_gt is not None:
        best_model = torch.load(best_model_files[best_idx])
        val_ccc_gt, _, _ = evaluate(best_model, data_loader_gt['devel'], params)
        test_ccc_gt, _, _ = evaluate(best_model, data_loader_gt['test'], params)
        print(f'On ground-truth labels:\t\t[Val CCC] for seed "{seeds[best_idx]}":\t{val_ccc_gt[0]:7.4f}')
        print(f'On ground-truth labels:\t\t[Test CCC] for seed "{seeds[best_idx]}":\t{test_ccc_gt[0]:7.4f}')
        print('-' * 100)

    # predict: val & test for best model
    if params.predict and data_loader_gt is None:
        print('Predict val & test videos...')
        best_model = torch.load(best_model_files[best_idx])
        
        ########################################
        if params.uncertainty_approach == None or params.uncertainty_approach == 'cw_ccc':
            predict(best_model, data_loader['test'], params)
        
        elif params.uncertainty_approach == "quantile_regression":
            train.predict_quantile_regression(best_model, data_loader['test'], params)
        
        elif params.uncertainty_approach == "monte_carlo_dropout":
            train.predict_mc_dropout(best_model, data_loader['test'], params)
        
        else:
            raise NotImplementedError()
        ########################################
        
        print('...done.')
    elif params.predict:  # data_loader_gt is available
        print('Predict train & val & test videos...')
        best_model = torch.load(best_model_files[best_idx])
        for k, v in data_loader_gt.items():
            predict(best_model, v, params)
        with open(os.path.join(config.PREDICTION_FOLDER, params.save_dir, "results.csv"), "a") as file:
            file.write(f"{params.preds_path},{best_mean_val_ccc:7.4f}\n")
        print('...done.')
    if not params.save:
        utils.delete_model(best_model_files[best_idx])


if __name__ == '__main__':
    # parse parameters
    params = parse_params()
    # register logger
    log_file_name = '{}_[{}]_[{}]_[{}]_[{}_{}_{}_{}]_[{}_{}_{}]_[{}_{}_{}_{}_{}]_{}.txt'.format(
        datetime.now(tz=tz.gettz('Asia/Shanghai')).strftime("%Y-%m-%d-%H-%M"),
        '_'.join(params.feature_set), '_'.join(params.emo_dim_set),
        'NOSEG' if not params.add_seg_id else 'SEG',
        params.rnn, params.d_rnn, params.rnn_n_layers, params.rnn_bi,
        params.attn, params.n_layers, params.n_heads,
        params.lr, params.batch_size, params.rnn_dr, params.attn_dr, params.out_dr, params.annotator
    )
    params.log_file_name = log_file_name
    if params.log:
        if not os.path.exists(config.LOG_FOLDER):
            os.makedirs(config.LOG_FOLDER)
        sys.stdout = utils.Logger(os.path.join(config.LOG_FOLDER, log_file_name))
    print(' '.join(sys.argv))

    main(params)
