# *_*coding:utf-8 *_*
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import utils


def train_model(model, data_loader, params):
    # data loader
    train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']
    # criterion
    if params.loss == 'ccc':
        criterion = utils.CCCLoss()
    elif params.loss == 'mse':
        criterion = utils.MSELoss()
    elif params.loss == 'l1':
        criterion = utils.L1Loss()
    elif params.loss == 'tilted':
        criterion = utils.TiltedLoss()
    elif params.loss == 'tiltedCCC':
        criterion = utils.TiltedCCCLoss()
    elif params.loss == 'cwCCC':
        criterion = utils.CCCLossWithStd()
    else:
        raise Exception(f'Not supported loss "{params.loss}".')
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.l2_penalty)
    # lr scheduler
    if params.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=params.lr_patience,
                                                 gamma=params.lr_factor)
    else:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                                            patience=params.lr_patience,
                                                            factor=params.lr_factor,
                                                            min_lr=params.min_lr,
                                                            verbose=True if params.log_extensive else False)
    # train
    best_val_loss = float('inf')
    best_val_ccc, best_val_pcc, best_val_rmse = [], [], []
    best_mean_val_ccc = -1
    best_model_file = ''
    early_stop = 0
    for epoch in range(1, params.epochs + 1):
        
        ################################
        if params.uncertainty_approach == 'cw_ccc':
            train_loss = train_with_std(model, train_loader, criterion, optimizer, epoch, params)
        
        else:
            train_loss = train(model, train_loader, criterion, optimizer, epoch, params)
        ################################
        if params.uncertainty_approach == "quantile_regression":
            val_loss, val_ccc, val_pcc, val_rmse = validate_quantile_regression(model, val_loader, criterion, params)
        
        elif params.uncertainty_approach == 'cw_ccc':
            val_loss, val_ccc, val_pcc, val_rmse = validate_std(model, val_loader, criterion, params)
        
        else:
            val_loss, val_ccc, val_pcc, val_rmse = validate(model, val_loader, criterion, params)
        ################################

        mean_val_ccc, mean_val_pcc, mean_val_rmse = np.mean(val_ccc), np.mean(val_pcc), np.mean(val_rmse)
        if params.log_extensive:
            print('-' * 50)
            print(f'Epoch:{epoch:>3} | [Train] | Loss: {train_loss:>.4f}')
            print(f'Epoch:{epoch:>3} |   [Val] | Loss: {val_loss:>.4f} | '
                  f'[CCC]: {mean_val_ccc:>7.4f} {[format(x, "7.4f") for x in val_ccc]} | '
                  f'PCC: {mean_val_pcc:>.4f} {[format(x, ".4f") for x in val_pcc]} | '
                  f'RMSE: {mean_val_rmse:>.4f} {[format(x, ".4f") for x in val_rmse]}')

        if mean_val_ccc > best_mean_val_ccc:
            best_val_ccc = val_ccc
            best_mean_val_ccc = np.mean(best_val_ccc)

            best_model_file = utils.save_model(model, params)
            if params.log_extensive:
                print(f'Epoch:{epoch:>3} | Save best model "{best_model_file}"!')
            best_val_loss, best_val_pcc, best_val_rmse = val_loss, val_pcc, val_rmse  # Note: loss, pcc and rmse when get best val ccc
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= params.early_stop:
                print(f'Note: target can not be optimized for {params.early_stop}'
                      f' consecutive epochs, early stop the training process!')
                break

        if params.lr_scheduler == 'step':
            lr_scheduler.step()
        else:
            lr_scheduler.step(1 - np.mean(val_ccc))

    best_mean_val_pcc, best_mean_val_rmse = np.mean(best_val_pcc), np.mean(best_val_rmse)

    print(f'Seed {params.current_seed} | '
          f'Best [Val CCC]:{best_mean_val_ccc:>7.4f} {[format(x, "7.4f") for x in best_val_ccc]}| '
          f'Loss: {best_val_loss:>.4f} | '
          f'PCC: {best_mean_val_pcc:>.4f} {[format(x, ".4f") for x in best_val_pcc]} | '
          f'RMSE: {best_mean_val_rmse:>.4f} {[format(x, ".4f") for x in best_val_rmse]}')

    return best_val_loss, best_val_ccc, best_val_pcc, best_val_rmse, best_model_file


def train(model, train_loader, criterion, optimizer, epoch, params):
    model.train()
    start_time = time.time()
    report_loss, report_size = 0, 0
    total_loss, total_size = 0, 0
    for batch, batch_data in enumerate(train_loader, 1):
        features, feature_lens, labels, metas = batch_data
        batch_size = features.size(0)
        # move to gpu if use gpu
        if params.gpu is not None:
            model.cuda()
            features = features.cuda()
            feature_lens = feature_lens.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        preds = model(features, feature_lens)
        # cal loss
        loss = 0.0
        for i in range(len(params.loss_weights)):
            
            if params.uncertainty_approach == "quantile_regression":
                branch_loss = criterion(preds, labels[:, :, i], feature_lens, params.label_smooth)
            
            else:
                branch_loss = criterion(preds[:, :, i], labels[:, :, i], feature_lens, params.label_smooth)

            loss = loss + params.loss_weights[i] * branch_loss
        loss.backward()
        if params.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.clip)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_size += batch_size

        report_loss += loss.item() * batch_size
        report_size += batch_size

        if batch % params.log_interval == 0 and params.log_extensive:
            avg_loss = report_loss / report_size
            elapsed_time = time.time() - start_time
            print(
                f"Epoch:{epoch:>3} | Batch: {batch:>3} | Lr: {optimizer.state_dict()['param_groups'][0]['lr']:>1.5f} | "
                f"Time used(s): {elapsed_time:>.1f} | Training loss: {avg_loss:>.4f}")
            report_loss, report_size, start_time = 0, 0, time.time()

    train_loss = total_loss / total_size
    return train_loss

def train_with_std(model, train_loader, criterion, optimizer, epoch, params):
    model.train()
    start_time = time.time()
    report_loss, report_size = 0, 0
    total_loss, total_size = 0, 0
    for batch, batch_data in enumerate(train_loader, 1):
        features, feature_lens, labels, metas, stdvs = batch_data
        batch_size = features.size(0)
        # move to gpu if use gpu
        if params.gpu is not None:
            model.cuda()
            features = features.cuda()
            feature_lens = feature_lens.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        preds = model(features, feature_lens)
        # cal loss
        loss = 0.0
        for i in range(len(params.loss_weights)):
            
            branch_loss = criterion(preds[:, :, i], labels[:, :, i], stdvs, feature_lens, params.label_smooth)

            loss = loss + params.loss_weights[i] * branch_loss
        loss.backward()
        if params.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.clip)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_size += batch_size

        report_loss += loss.item() * batch_size
        report_size += batch_size

        if batch % params.log_interval == 0 and params.log_extensive:
            avg_loss = report_loss / report_size
            elapsed_time = time.time() - start_time
            print(
                f"Epoch:{epoch:>3} | Batch: {batch:>3} | Lr: {optimizer.state_dict()['param_groups'][0]['lr']:>1.5f} | "
                f"Time used(s): {elapsed_time:>.1f} | Training loss: {avg_loss:>.4f}")
            report_loss, report_size, start_time = 0, 0, time.time()

    train_loss = total_loss / total_size
    return train_loss

def validate(model, val_loader, criterion, params):
    model.eval()
    full_preds, full_labels = [], []
    with torch.no_grad():
        val_loss = 0
        val_size = 0
        for batch, batch_data in enumerate(val_loader, 1):
            
            features, feature_lens, labels, _ = batch_data
            
            batch_size = features.size(0)
            # move to gpu if use gpu
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
            preds = model(features, feature_lens)
            # cal loss
            loss = 0.0
            for i in range(len(params.loss_weights)):
                
                branch_loss = criterion(preds[:, :, i], labels[:, :, i], feature_lens, params.label_smooth)

                loss = loss + params.loss_weights[i] * branch_loss
            val_loss += loss.item() * batch_size
            val_size += batch_size

            full_preds.append(preds.cpu().detach().squeeze(0).numpy())
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())
        val_loss /= val_size
        val_ccc, val_pcc, val_rmse = utils.eval(full_preds, full_labels)

    return val_loss, val_ccc, val_pcc, val_rmse

def validate_std(model, val_loader, criterion, params):
    model.eval()
    full_preds, full_labels = [], []
    with torch.no_grad():
        val_loss = 0
        val_size = 0
        for batch, batch_data in enumerate(val_loader, 1):
            
            features, feature_lens, labels, _, std = batch_data# NOTE: with std
            
            batch_size = features.size(0)
            # move to gpu if use gpu
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
            preds = model(features, feature_lens)
            # cal loss
            loss = 0.0
            for i in range(len(params.loss_weights)):
                
                branch_loss = utils.CCCLoss()(preds[:, :, i], labels[:, :, i], feature_lens, params.label_smooth)

                loss = loss + params.loss_weights[i] * branch_loss
            val_loss += loss.item() * batch_size
            val_size += batch_size

            full_preds.append(preds.cpu().detach().squeeze(0).numpy())
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())
        val_loss /= val_size
        val_ccc, val_pcc, val_rmse = utils.eval(full_preds, full_labels)

    return val_loss, val_ccc, val_pcc, val_rmse


def evaluate(model, test_loader, params):
    model.eval()
    full_preds, full_labels = [], []
    with torch.no_grad():
        for batch, batch_data in enumerate(test_loader, 1):
            features, feature_lens, labels, meta = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
            preds = model(features, feature_lens)
            full_preds.append(preds.cpu().detach().squeeze(0).numpy())
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())

        test_ccc, test_pcc, test_rmse = utils.eval(full_preds, full_labels)
    return test_ccc, test_pcc, test_rmse


def predict(model, data_loader, params):
    model.eval()
    full_preds, full_metas, full_labels = [], [], []
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, feature_lens, labels, metas = batch_data
            # move to gpu if use gpu
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
            preds = model(features, feature_lens)
            full_preds.append(preds.cpu().detach().squeeze(0).numpy())
            full_metas.append(metas.detach().squeeze(0).numpy())
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())

        partition = data_loader.dataset.partition
        utils.write_model_prediction(full_metas, full_preds, full_labels, params, partition=partition, view=params.view)



import os
import config
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def validate_quantile_regression(model, val_loader, criterion, params):
    model.eval()
    full_preds, full_labels = [], []
    with torch.no_grad():
        val_loss = 0
        val_size = 0
        for batch, batch_data in enumerate(val_loader, 1):
            features, feature_lens, labels, _ = batch_data
            batch_size = features.size(0)
            # move to gpu if use gpu
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
            preds = model(features, feature_lens)
            # cal loss
            loss = 0.0
            for i in range(len(params.loss_weights)):
                
                # branch_loss = criterion(preds[:, :, i], labels[:, :, i], feature_lens, params.label_smooth)
                branch_loss = criterion(preds, labels[:, :, i], feature_lens, params.label_smooth)# NOTE: for tilted loss

                loss = loss + params.loss_weights[i] * branch_loss
            val_loss += loss.item() * batch_size
            val_size += batch_size

            # print("Preds:", preds.cpu().detach().squeeze(0).numpy().shape)
            # print("Labels:", labels.cpu().detach().squeeze(0).numpy().shape)

            # preds = preds.cpu().detach().squeeze(0).numpy()# NOTE: for non-tilted-losses
            
            # preds = preds[:, :, 1:2]# NOTE: use 0.5 quantile for prediction
            
            preds = preds.cpu().detach().squeeze(0).numpy()
            preds = preds.mean(axis=1).reshape((preds.shape[0], 1))# NOTE: use mean of all quantiles as prediction

            full_preds.append(preds)
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())
        val_loss /= val_size
        val_ccc, val_pcc, val_rmse = utils.eval(full_preds, full_labels)

    return val_loss, val_ccc, val_pcc, val_rmse

def evaluate_quantile_regression(model, test_loader, params):
    model.eval()
    full_preds, full_labels = [], []
    with torch.no_grad():
        for batch, batch_data in enumerate(test_loader, 1):
            features, feature_lens, labels, meta = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
            
            preds = model(features, feature_lens)
            # preds = preds[:, :, 1:2]# NOTE: use 0.5 quantile for prediction
            preds = preds.mean(axis=2).reshape((preds.shape[0], preds.shape[1], 1))# NOTE: use mean of all quantiles as prediction
            
            full_preds.append(preds.cpu().detach().squeeze(0).numpy())
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())

        test_ccc, test_pcc, test_rmse = utils.eval(full_preds, full_labels)
    return test_ccc, test_pcc, test_rmse

def predict_quantile_regression(model, data_loader, params):
    with torch.no_grad():

        full_preds, full_metas, full_labels = [], [], []
        for _, batch_data in enumerate(data_loader, 1):
            features, feature_lens, labels, metas = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
            preds = model(features, feature_lens)
            full_preds.append(preds.cpu().detach().squeeze(0).numpy())
            full_metas.append(metas.detach().squeeze(0).numpy())
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())

        prediction_folder = config.PREDICTION_FOLDER
        if params.save_dir is not None:
            prediction_folder = os.path.join(prediction_folder, params.save_dir)
        if not os.path.exists(prediction_folder):
            os.makedirs(prediction_folder)
        folder = f'{os.path.splitext(params.log_file_name)[0]}_[{params.n_seeds}_{params.current_seed}]_quantile_regression'
        save_dir = os.path.join(prediction_folder, folder)
        params.preds_path = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_dir = os.path.join(save_dir, 'img')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        
        partition = data_loader.dataset.partition

        # full_preds = np.row_stack(full_preds)
        # full_labels = np.row_stack(full_labels)
        # full_metas = np.row_stack(full_metas)
        # print("full_preds:", full_preds.shape)
        # print("full_labels:", full_labels.shape)
        for idx, emo_dim in enumerate(params.emo_dim_set):
            for i in range(len(full_preds)):
                meta = full_metas[i]
                label = full_labels[i]
                vid = meta[0, 0]

                pred = full_preds[i]

                # print("pred:", pred.shape)

                pred_q0 = pred[:, 0]
                pred_q1 = pred[:, 1]
                pred_q2 = pred[:, 2]

                img_emo_dir = os.path.join(img_dir, emo_dim)
                if not os.path.exists(img_emo_dir):
                    os.mkdir(img_emo_dir)
                    
                plot_video_prediction_with_quantiles(meta[:, 1], pred_q0, pred_q1, pred_q2, label, partition, vid, emo_dim, img_emo_dir)

def plot_video_prediction_with_quantiles(time, pred_q0, pred_q1, pred_q2, label_raw, partition, vid, emo_dim, save_dir):

    n = 100
    time = time[:n]
    
    pred_q0 = pred_q0[:n]
    pred_q1 = pred_q1[:n]
    pred_q2 = pred_q2[:n]

    # df_pred = df_pred[df_pred['segment_id'] > 0]  # remove padding

    time = time / 1000.0

    # label_raw = [item for sublist in label_raw for item in sublist]
    label_raw = label_raw[:len(time)]
    # label_target = savgol_filter(label_raw, 11, 3).tolist()
    label_target = label_raw

    plt.figure(figsize=(20, 10))

    plt.plot(time, label_target, 'red', label='target')
    
    plt.fill_between(time, pred_q0, pred_q2, color='lightblue', alpha=.5)
    plt.plot(time, pred_q0, 'blue', label=r"$\tau=0.1$")
    plt.plot(time, pred_q2, 'green', label=r"$\tau=0.9$")

    plt.plot(time, pred_q1, 'orange', label=r"$\tau=0.5$")

    plt.title(f"{emo_dim} of video '{vid}' [{partition}]", fontsize=24)
    plt.legend(prop={"size": 24})
    plt.xlabel('time (s)', fontsize=24)
    plt.ylabel(emo_dim, fontsize=24)

    ax = plt.gca()
    if time[-1] < 400:
        x_interval = 10
    elif time[-1] < 800:
        x_interval = 20
    else:
        x_interval = 50
    x_major_locator = plt.MultipleLocator(x_interval)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim([-1, 1])
    plt.grid()

    plt.savefig(os.path.join(save_dir, f'{vid}.jpg'))
    plt.close()

def evaluate_mc_dropout(model, test_loader, params):
    n_ensembles = 10
    model.train()
    full_preds, full_labels = [], []
    with torch.no_grad():
        for batch, batch_data in enumerate(test_loader, 1):
            features, feature_lens, labels, meta = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
            preds = [model(features, feature_lens).cpu().detach().squeeze(0).numpy() for i in range(n_ensembles)]
            # print(np.array(preds).shape)
            preds = np.mean(preds, axis=0)# TODO check
            full_preds.append(preds)
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())

        test_ccc, test_pcc, test_rmse = utils.eval(full_preds, full_labels)
    return test_ccc, test_pcc, test_rmse

def predict_mc_dropout(model, data_loader, params):
    model.train()
    n_ensembles = 4

    full_preds, full_metas, full_labels = [], [], []
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, feature_lens, labels, metas = batch_data
            # move to gpu if use gpu
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
            preds = [model(features, feature_lens).cpu().detach().squeeze(0).numpy() for i in range(n_ensembles)]
            full_preds.append(np.array(preds))
            full_metas.append(metas.detach().squeeze(0).numpy())
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())

        prediction_folder = config.PREDICTION_FOLDER
        if params.save_dir is not None:
            prediction_folder = os.path.join(prediction_folder, params.save_dir)
        if not os.path.exists(prediction_folder):
            os.makedirs(prediction_folder)
        folder = f'{os.path.splitext(params.log_file_name)[0]}_[{params.n_seeds}_{params.current_seed}]_mc_dropout'
        save_dir = os.path.join(prediction_folder, folder)
        params.preds_path = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_dir = os.path.join(save_dir, 'img')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        
        partition = data_loader.dataset.partition

        for idx, emo_dim in enumerate(params.emo_dim_set):
            for i in range(len(full_preds)):

                meta = full_metas[i]
                vid = meta[0, 0]
                label = full_labels[i]
                
                pred = full_preds[i]

                pred = pred.squeeze(2)
                pred_mean = np.mean(pred, axis=0)
                pred_std = np.std(pred, axis=0)
                pred_min = np.min(pred, axis=0)
                pred_max = np.max(pred, axis=0)

                # print(pred.shape)

                img_emo_dir = os.path.join(img_dir, emo_dim)
                if not os.path.exists(img_emo_dir):
                    os.mkdir(img_emo_dir)
                    
                plot_video_prediction_with_uncertainty(meta[:, 1], pred_mean, pred_std, label, partition, vid, emo_dim, img_emo_dir, pred_min, pred_max)


def plot_video_prediction_with_uncertainty(time, pred_mean, pred_var, label_raw, partition, vid, emo_dim, save_dir, pred_min, pred_max):

    n = 100
    time = time[:n]
    pred_mean = pred_mean[:n]
    pred_var = pred_var[:n]
    pred_min = pred_min[:n]
    pred_max = pred_max[:n]

    # df_pred = df_pred[df_pred['segment_id'] > 0]  # remove padding

    time = time / 1000.0

    # label_raw = [item for sublist in label_raw for item in sublist]
    label_raw = label_raw[:len(time)]
    # label_target = savgol_filter(label_raw, 11, 3).tolist()
    label_target = label_raw

    plt.figure(figsize=(20, 10))
    
    # plt.fill_between(time, pred_min, pred_max, color='lightblue', alpha=.5)
    plt.fill_between(time, pred_mean - pred_var, pred_mean + pred_var, color='lightblue', alpha=.5)

    plt.plot(time, label_target, 'red', label='target')
    plt.plot(time, pred_mean, 'blue', label=f'prediction')

    plt.title(f"{emo_dim} of video '{vid}' [{partition}]", fontsize=24)
    plt.legend(prop={"size": 24})
    plt.xlabel('time', fontsize=24)
    plt.ylabel(emo_dim, fontsize=24)

    ax = plt.gca()
    if time[-1] < 400:
        x_interval = 10
    elif time[-1] < 800:
        x_interval = 20
    else:
        x_interval = 50
    x_major_locator = plt.MultipleLocator(x_interval)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim([-1, 1])
    plt.grid()

    plt.savefig(os.path.join(save_dir, f'{vid}.jpg'))
    plt.close()