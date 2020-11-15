# *_*coding:utf-8 *_*
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import torch.nn as nn
import json
import config
from sklearn import preprocessing
from scipy.signal import savgol_filter


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


def load_data(params, feature_set, emo_dim_set,
              normalize=True,
              label_preproc=None,
              norm_opts=None,
              segment_type='normal',
              win_len=100, hop_len=100,
              feature_path=config.PATH_TO_ALIGNED_FEATURES,
              label_path=config.PATH_TO_LABELS,
              save=False,
              refresh=False,
              add_seg_id=False,
              annotator=None):
    data_file_name = '_'.join(
        feature_set + emo_dim_set) + f'_{normalize}_{add_seg_id}_{segment_type}_{win_len}_{hop_len}_{annotator}.pkl'
    data_file = os.path.join(config.DATA_FOLDER, data_file_name)
    if os.path.exists(data_file) and not refresh:
        print(f'Find cached data "{os.path.basename(data_file)}".')
        data = pickle.load(open(data_file, 'rb'))
        if label_preproc is not None and annotator is not None:
            data = preprocess_labels(data, params)
        return data

    if annotator is not None:
        unique_anno_mapping = get_unique_annotator_mapping(config.ANNOTATOR_MAPPING)
        label_path = config.PATH_TO_LABELS_RAW
        anno2vid_file_name = 'anno_2_vid_{}.pkl'.format(emo_dim_set[0])
        anno2vid_file = os.path.join(config.DATA_FOLDER, anno2vid_file_name)
        if os.path.exists(anno2vid_file):
            print('Found cached annotator 2 video mapping.')
            anno2vid = pickle.load(open(anno2vid_file, 'rb'))
        else:
            anno2vid, _ = get_anno_vid_mapping(emo_dim_set[0])
            print('Dumping anno2vid mapping...')
            pickle.dump(anno2vid, open(anno2vid_file, 'wb'))

    print('Constructing data from scratch ...')
    data = {'train': {'feature': [], 'label': [], 'meta': []},
            'devel': {'feature': [], 'label': [], 'meta': []},
            'test': {'feature': [], 'label': [], 'meta': []}}
    vid2partition, partition2vid = get_data_partition(config.PARTITION_FILE)
    feature_dims = [0] * len(feature_set)
    if add_seg_id:
        feature_idx = 1
        print(f'Note: add segment id in the feature.')
    else:
        feature_idx = 2

    for partition, vids in partition2vid.items():
        if annotator is not None:
            vids = [vid for vid in anno2vid[annotator] if vid in vids]
        for vid in vids:
            # concat
            sample_concat_data = []  # feature1, feature2, ..., emo dim1, emo dim2. (ex, 'au', 'vggface', 'arousal', 'valence')
            ## feature
            for i, feature in enumerate(feature_set):
                feature_file = os.path.join(feature_path, feature, vid + '.csv')
                assert os.path.exists(feature_file), f'Error: no available "{feature}" feature file for video "{vid}".'
                df = pd.read_csv(feature_file)
                feature_dims[i] = df.shape[1] - 2
                if i == 0:
                    feature_data = df  # keep timestamp and segment id in 1st feature val
                else:
                    feature_data = df.iloc[:, 2:]  # feature val starts from third column
                sample_concat_data.append(feature_data)
            data[partition]['feature_dims'] = feature_dims
            ## label
            for emo_dim in emo_dim_set:
                label_file = os.path.join(label_path, emo_dim, vid + '.csv')
                assert os.path.exists(label_file), f'Error: no available "{emo_dim}" label file for video "{vid}".'
                df = pd.read_csv(label_file)
                if annotator is not None:
                    a_id = annotator
                    if a_id not in df.columns:
                        try:
                            a_id = [i for i in df.columns if i.isdigit() and unique_anno_mapping[i] == a_id][0]
                        except IndexError:
                            print('Annotator {} not available for this video ({})'.format(annotator, df.columns))
                    label = df[a_id] / 1000  # [-1000, 1000] -> [-1, 1]
                    label = label.values
                else:
                    label = df['value'].values
                label_data = pd.DataFrame(data=label, columns=[emo_dim])
                sample_concat_data.append(label_data)

            # concat
            sample_concat_data = pd.concat(sample_concat_data, axis=1)
            sample_concat_data = sample_concat_data.dropna()
            # segment train samples, NOTE: do not segment devel and test samples!
            if partition == 'train' and segment_type != 'None':
                samples = segment_sample(sample_concat_data, segment_type, win_len, hop_len)  # segmented samples: list
            else:
                samples = [sample_concat_data]
            # store
            for i, segment in enumerate(samples):
                meta = np.column_stack((np.array([int(vid)] * len(segment)),
                                        segment.iloc[:, :2].values))  # video id, time stamp, segment id
                data[partition]['meta'].append(meta)
                data[partition]['label'].append(segment.iloc[:, -len(emo_dim_set):].values)
                data[partition]['feature'].append(segment.iloc[:, feature_idx:-len(
                    emo_dim_set)].values)  # feature val starts from the "feature_idx"th column

    if normalize:  # mainly for audio features
        idx_list = []
        if add_seg_id:  # norm seg id
            feature_dims = [1] + feature_dims
            feature_set = ['seg_id'] + feature_set

        assert norm_opts is not None and len(norm_opts) == len(feature_set)
        norm_opts = [True if norm_opt == 'y' else False for norm_opt in norm_opts]
        print('Feature dims: ', feature_dims)
        feature_dims = np.cumsum(feature_dims).tolist()
        feature_dims = [0] + feature_dims
        feature_idxs = zip(feature_dims[0:-1], feature_dims[1:])
        norm_feature_set = []
        for i, (s_idx, e_idx) in enumerate(feature_idxs):
            norm_opt, feature = norm_opts[i], feature_set[i]
            if norm_opt:
                norm_feature_set.append(feature)
                idx_list.append([s_idx, e_idx])
        print('Normalize features: ', norm_feature_set)
        print('Indices of normalized features: ', idx_list)
        data = normalize_data(data, idx_list)
    # save data
    if save:
        print('Dumping data...')
        pickle.dump(data, open(data_file, 'wb'))
    if label_preproc is not None and annotator is not None:
        data = preprocess_labels(data, params)

    return data


def preprocess_labels(data, params):
    train_concat_data = np.row_stack(data['train']['label'])
    train_mean = np.nanmean(train_concat_data, axis=0)
    train_std = np.nanstd(train_concat_data, axis=0)

    if params.label_preproc == 'standard':
        for part in data.keys():
            data[part]['label'] = (data[part]['label'] - train_mean) / train_std
        max_val = max([np.max(arr) for partition in data.keys() for arr in data[partition]['label']])
        min_val = min([np.min(arr) for partition in data.keys() for arr in data[partition]['label']])
        print('Standardized labels (min {}, max {}).'.format(min_val, max_val))
    elif params.label_preproc == 'norm':
        for part in data.keys():
            for row in range(len(data[part]['label'])):
                data[part]['label'][row] = preprocessing.normalize(data[part]['label'][row])
        max_val = max([np.max(arr) for partition in data.keys() for arr in data[partition]['label']])
        min_val = min([np.min(arr) for partition in data.keys() for arr in data[partition]['label']])
        print('Normalized labels (min {}, max {}).'.format(min_val, max_val))
    elif params.label_preproc == 'savgol':
        for part in data.keys():
            for row in range(len(data[part]['label'])):
                labels = np.hstack(data[part]['label'][row])
                filtered_labels = savgol_filter(labels, params.savgol_window, params.savgol_polyorder).tolist()
                data[part]['label'][row] = [[value] for value in filtered_labels]
        max_val = max([np.max(arr) for partition in data.keys() for arr in data[partition]['label']])
        min_val = min([np.min(arr) for partition in data.keys() for arr in data[partition]['label']])
        print('Applied Savitzky-Golay Filter to labels (min {}, max {}).'.format(min_val, max_val))
    elif params.label_preproc == 'None':
        max_val = max([np.max(arr) for partition in data.keys() for arr in data[partition]['label']])
        min_val = min([np.min(arr) for partition in data.keys() for arr in data[partition]['label']])
        print('No label preprocessing (min {}, max {}).'.format(min_val, max_val))
    else:
        print('Label preprocessing {} not implemented.'.format(params.label_preproc))

    return data


def load_fusion_data(pred_dirs, emo_dim_set,
                     segment_type=None,
                     win_len=200, hop_len=100,
                     normalize=False,
                     label_path=config.PATH_TO_LABELS):
    print('Constructing fusion data from scratch ...')
    data = {'train': {'feature': [], 'label': [], 'meta': []},
            'devel': {'feature': [], 'label': [], 'meta': []},
            'test': {'feature': [], 'label': [], 'meta': []}}

    vid2partition, partition2vid = get_data_partition(config.PARTITION_FILE)
    for partition, vids in partition2vid.items():
        for vid in vids:
            # concat
            sample_concat_data = []  # pred_1 emo_dim_1, pred_1 emo_dim_2, ..., label emo_dim_1, label emo_dim_2.
            ## preds
            first = True
            # annos for this vid, check if pred_dir is in annos
            for pred_dir in pred_dirs:
                for emo_dim in emo_dim_set:  # concat emo dim
                    pred_file = os.path.join(pred_dir, f'csv/{emo_dim}/{vid}.csv')
                    if os.path.exists(pred_file):
                        df = pd.read_csv(pred_file)
                    elif os.path.exists(os.path.join(pred_dir, f'csv/{emo_dim}/{vid}.0.csv')):
                        df = pd.read_csv(os.path.join(pred_dir, f'csv/{emo_dim}/{vid}.0.csv'))
                    else:
                        print('File not found {}.'.format(pred_file))
                    if first:
                        cols = list(df)  # timestamp, value, segment_id
                        cols[1], cols[2] = cols[2], cols[1]  # exchange value and segment_id
                        feature_data = df.loc[:, cols]  # keep timestamp and segment id in 1st feature val
                        first = False
                    else:
                        feature_data = df.iloc[:, 1]  # prediction value in second column
                    sample_concat_data.append(feature_data)
            ## label
            for emo_dim in emo_dim_set:
                label_file = os.path.join(label_path, emo_dim, vid + '.csv')
                assert os.path.exists(label_file), f'Error: no available "{emo_dim}" label file for video "{vid}".'
                df = pd.read_csv(label_file)
                label_data = df.iloc[:, [1]].rename(columns={'value': emo_dim})  # label value is in second column
                sample_concat_data.append(label_data)
            # concat
            sample_concat_data = pd.concat(sample_concat_data, axis=1)
            sample_concat_data = sample_concat_data.reset_index(drop=True)
            sample_concat_data = sample_concat_data.dropna()
            # segment train samples, NOTE: do not segment devel and test samples!
            if partition == 'train' and segment_type is not None:
                samples = segment_sample(sample_concat_data, segment_type, win_len, hop_len)  # segmented samples: list
            else:
                samples = [sample_concat_data]
            # store
            for i, segment in enumerate(samples):
                meta = np.column_stack((np.array([int(vid)] * len(segment)),
                                        segment.iloc[:, :2].values))  # video id, time stamp, segment id
                data[partition]['meta'].append(meta)
                data[partition]['feature'].append(segment.iloc[:, 2:-len(emo_dim_set)].values)
                data[partition]['label'].append(segment.iloc[:, -len(emo_dim_set):].values)

    if normalize:
        input_dim = data['train']['feature'][0].shape[1]
        assert input_dim == len(emo_dim_set) * len(pred_dirs)
        idx_list = [0, input_dim]
        data = normalize_data(data, idx_list)
    return data


def normalize_data(data, idx_list, column_name='feature'):
    if len(idx_list) == 0:  # modified
        return data
    train_concat_data = np.row_stack(data['train'][column_name])
    train_mean = np.nanmean(train_concat_data, axis=0)
    train_std = np.nanstd(train_concat_data, axis=0)

    for partition in data.keys():
        for i in range(len(data[partition][column_name])):
            for s_idx, e_idx in idx_list:
                data[partition][column_name][i][:, s_idx:e_idx] = \
                    (data[partition][column_name][i][:, s_idx:e_idx] - train_mean[s_idx:e_idx]) / (
                            train_std[s_idx:e_idx] + config.EPSILON)
                data[partition][column_name][i][:, s_idx:e_idx] = np.where(  # get rid of any nans
                    np.isnan(data[partition][column_name][i][:, s_idx:e_idx]), 0.0,
                    data[partition][column_name][i][:, s_idx:e_idx])

    return data


def segment_sample(sample, segment_type, win_len, hop_len=None, is_training=False):
    segmented_sample = []
    if hop_len is None:
        hop_len = win_len
    else:
        assert hop_len <= win_len
    if segment_type == 'id':
        segment_ids = sorted(set(sample['segment_id'].values))
        for id in segment_ids:
            segment = sample[sample['segment_id'] == id]
            for s_idx in range(0, len(segment), hop_len):
                e_idx = min(s_idx + win_len, len(segment))
                sub_segment = segment.iloc[s_idx:e_idx]
                segmented_sample.append(sub_segment)
                if e_idx == len(segment):
                    break
    elif segment_type == 'normal':
        for s_idx in range(0, len(sample), hop_len):
            e_idx = min(s_idx + win_len, len(sample))
            # s_idx_ = max(0, len(sample) - win_len) if e_idx == len(sample) else s_idx  # added: 07/07
            if (e_idx - s_idx) < 20:
                print('Warning: encounter too short segment with length less than 20.')
            segment = sample.iloc[s_idx:e_idx]
            segmented_sample.append(segment)
            if e_idx == len(sample):
                break
    else:
        raise Exception(f'Not supported segment type "{segment_type}" to segment.')
    return segmented_sample


# video id (ex, '23') <--> data partition (ex, 'train')
def get_data_partition(partition_file):
    vid2partition, partition2vid = {}, {}
    df = pd.read_csv(partition_file)

    for row in df.values:
        vid, partition = str(row[0]), row[-1]  # row[2]  #  # video id is string
        vid2partition[vid] = partition
        if partition not in partition2vid:
            partition2vid[partition] = []
        if vid not in partition2vid[
            partition]:  # Note: this is necessary because few items repeat 2 times in partition file.
            partition2vid[partition].append(vid)
    return vid2partition, partition2vid


def get_unique_annotator_mapping(mapping_file):
    with open(mapping_file) as json_file:
        mapping = json.load(json_file)
    return mapping


def get_anno_vid_mapping(emo_dim):
    unique_annotators = get_unique_annotator_mapping(config.ANNOTATOR_MAPPING)
    raw_label_path = os.path.join(config.PATH_TO_LABELS_RAW, emo_dim)
    anno2vid, vid2anno = {}, {}

    for vid in os.listdir(raw_label_path):
        vid_id = vid[:-4]  # .csv
        cols = pd.read_csv(os.path.join(raw_label_path, vid), nrows=0).columns.tolist()
        annos = cols[2:-1]  # e.g., timestamp,segment_id,1,2,3,4,7,label_arousal
        unique_annos = [unique_annotators[anno] for anno in annos]
        vid2anno[vid_id] = unique_annos

    for k, value in vid2anno.items():
        for v in value:
            anno2vid.setdefault(int(v), []).append(k)
    return anno2vid, vid2anno


def get_padding_mask(x, x_lens):
    """
    :param x: (seq_len, batch_size, feature_dim)
    :param x_lens: sequence lengths within a batch with size (batch_size,)
    :return: padding_mask with size (batch_size, seq_len)
    """
    seq_len, batch_size, feature_dim = x.size()
    mask = torch.ones(batch_size, seq_len, device=x.device)
    for seq, seq_len in enumerate(x_lens):
        mask[seq, :seq_len] = 0
    mask = mask.bool()
    return mask


class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true, seq_lens=None, label_smooth=None):
        """
        :param y_pred: (batch_size, seq_len)
        :param y_true: (batch_size, seq_len)
        :param seq_lens: (batch_size,)
        :return:
        """
        # make padding mask
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
        else:
            mask = torch.ones_like(y_true, device=y_true.device)
        # smooth label by average pooling
        if label_smooth is not None:
            y_true = torch.nn.functional.avg_pool1d(y_true.unsqueeze(1), kernel_size=label_smooth,
                                                    stride=1, padding=(label_smooth - 1) // 2,
                                                    count_include_pad=False)
            y_true = y_true.squeeze(1)

        y_true_mean = torch.sum(y_true * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        y_pred_mean = torch.sum(y_pred * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        # biased variance
        y_true_var = torch.sum(mask * (y_true - y_true_mean) ** 2, dim=1, keepdim=True) / torch.sum(mask, dim=1,
                                                                                                    keepdim=True)
        y_pred_var = torch.sum(mask * (y_pred - y_pred_mean) ** 2, dim=1, keepdim=True) / torch.sum(mask, dim=1,
                                                                                                    keepdim=True)

        cov = torch.sum(mask * (y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1, keepdim=True) / torch.sum(mask,
                                                                                                                 dim=1,
                                                                                                                 keepdim=True)

        ccc = torch.mean(2.0 * cov / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2), dim=0)  # (1,*)
        ccc = ccc.squeeze(0)  # (*,) if necessary
        ccc_loss = 1.0 - ccc

        return ccc_loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true, seq_lens=None, label_smooth=None):
        """
        :param y_pred: (batch_size, seq_len)
        :param y_true: (batch_size, seq_len)
        :return:
        """
        # smooth label by average pooling
        if label_smooth is not None:
            y_true = torch.nn.functional.avg_pool1d(y_true.unsqueeze(1), kernel_size=label_smooth,
                                                    stride=1, padding=(label_smooth - 1) // 2,
                                                    count_include_pad=False)
            y_true = y_true.squeeze(1)

        # get mask
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
            loss = torch.nn.functional.mse_loss(y_pred, y_true, reduction='none')
            # loss = loss * mask
            # loss = loss.sum() / seq_lens.sum()
            mask = mask.bool()
            loss = loss.masked_select(mask)
            loss = loss.mean()
        else:
            loss = torch.nn.functional.mse_loss(y_pred, y_true)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, y_pred, y_true, seq_lens=None, label_smooth=None):
        """
        :param y_pred: (batch_size, seq_len)
        :param y_true: (batch_size, seq_len)
        :return:
        """
        # smooth label by average pooling
        if label_smooth is not None:
            y_true = torch.nn.functional.avg_pool1d(y_true.unsqueeze(1), kernel_size=label_smooth,
                                                    stride=1, padding=(label_smooth - 1) // 2,
                                                    count_include_pad=False)
            y_true = y_true.squeeze(1)

        # get mask
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
            loss = torch.nn.functional.l1_loss(y_pred, y_true, reduction='none')
            # loss = loss * mask
            # loss = loss.sum() / seq_lens.sum()
            mask = mask.bool()
            loss = loss.masked_select(mask)
            loss = loss.mean()
        else:
            loss = torch.nn.functional.l1_loss(y_pred, y_true)
        return loss


def eval(full_preds, full_labels):
    full_preds = np.row_stack(full_preds)
    full_labels = np.row_stack(full_labels)
    assert full_preds.shape == full_labels.shape
    n_targets = full_preds.shape[1]
    val_ccc, val_pcc, val_rmse = [], [], []
    for i in range(n_targets):
        preds = full_preds[:, i]
        labels = full_labels[:, i]
        ccc, pcc, rmse = cal_eval_metrics(preds, labels)
        val_ccc.append(ccc)
        val_pcc.append(pcc)
        val_rmse.append(rmse)
    return val_ccc, val_pcc, val_rmse


# ccc, pcc, mse
def cal_eval_metrics(preds, labels):
    rmse = np.sqrt(np.mean((preds - labels) ** 2))

    preds_mean, labels_mean = np.mean(preds), np.mean(labels)
    cov_mat = np.cov(preds, labels)  # Note: unbiased
    covariance = cov_mat[0, 1]
    preds_var, labels_var = cov_mat[0, 0], cov_mat[1, 1]

    pcc = covariance / np.sqrt(preds_var * labels_var)
    ccc = 2.0 * covariance / (preds_var + labels_var + (preds_mean - labels_mean) ** 2)
    return ccc, pcc, rmse


def save_model(model, params):
    model_dir = os.path.join(config.MODEL_FOLDER)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if params.label_preproc == 'savgol':
        preproc_args = '_'.join([str(params.label_preproc), str(params.savgol_window), str(params.savgol_polyorder)])
    else:
        preproc_args = str(params.label_preproc)
    if os.path.splitext(params.log_file_name)[0].split('_')[1] != '[FUSION]':
        arg = str(params.annotator) + '_' + preproc_args
    else:
        arg = preproc_args
    model_file_name = f'{os.path.splitext(params.log_file_name)[0]}_[{params.n_seeds}_{params.current_seed}_{arg}].pth'
    model_file = os.path.join(model_dir, model_file_name)
    torch.save(model, model_file)

    return model_file


def delete_model(model_file):
    if os.path.exists(model_file):
        os.remove(model_file)
        print(f'Delete model "{model_file}".')
    else:
        print(f'Warning: model file "{model_file}" does not exist when delete it!')


def write_model_prediction(metas, preds, labels, params, partition, view=False):
    """
    :param metas: # video id, time stamp, segment id
    :param preds:
    :param params:
    :param partition:
    :param view: whether plot predicted arousal and valence or not
    :return:
    """
    # write prediction sample by sample (multiple files)
    prediction_folder = config.PREDICTION_FOLDER
    if params.save_dir is not None:
        prediction_folder = os.path.join(prediction_folder, params.save_dir)
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)

    folder = f'{os.path.splitext(params.log_file_name)[0]}_[{params.n_seeds}_{params.current_seed}]'
    save_dir = os.path.join(prediction_folder, folder)
    params.preds_path = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    csv_dir = os.path.join(save_dir, 'csv')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    if view:
        img_dir = os.path.join(save_dir, 'img')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

    for idx, emo_dim in enumerate(params.emo_dim_set):
        csv_emo_dir = os.path.join(csv_dir, emo_dim)
        if not os.path.exists(csv_emo_dir):
            os.mkdir(csv_emo_dir)
        columns = ['timestamp', 'value', 'segment_id']
        for meta, pred, label in zip(metas, preds, labels):
            vid = meta[0, 0]
            # csv
            sample_file_name = f'{int(vid)}.csv'  # [vid].csv, ex: 1.csv
            sample_data = np.column_stack([meta[:, 1], pred[:, idx], meta[:, 2]])
            df = pd.DataFrame(sample_data, columns=columns)
            df[['timestamp', 'segment_id']] = df[['timestamp', 'segment_id']].astype(np.int)
            sample_pred_file = os.path.join(csv_emo_dir, sample_file_name)
            df.to_csv(sample_pred_file, index=False)

            # plot img
            if view:
                img_emo_dir = os.path.join(img_dir, emo_dim)
                if not os.path.exists(img_emo_dir):
                    os.mkdir(img_emo_dir)
                plot_video_prediction(df, label, partition, vid, emo_dim, img_emo_dir)

    # write aggregated prediction (all in one file)
    metas = np.row_stack(metas)
    metas = metas[:, :2]
    preds = np.row_stack(preds)
    data = np.column_stack([metas, preds])
    columns = ['id', 'timestamp'] + ['prediction_' + emo_dim for emo_dim in params.emo_dim_set]
    df = pd.DataFrame(data, columns=columns)
    df[['id', 'timestamp']] = df[['id', 'timestamp']].astype(np.int)
    pred_file_name = f'{partition}.csv'
    aggr_dir = os.path.join(csv_dir, 'aggregated')
    if not os.path.exists(aggr_dir):
        os.mkdir(aggr_dir)
    pred_file = os.path.join(aggr_dir, pred_file_name)
    if os.path.exists(pred_file):
        df_existed = pd.read_csv(pred_file)
        cols_existed = list(df_existed)
        cols = list(df)
        assert len(cols) == 3 and len(cols_existed) == 3 and (cols[-1] != cols_existed[-1]), \
            f'Error: cannot merge existed prediction file "{pred_file}".'
        df = pd.merge(df, df_existed) if cols[-1] == 'prediction_arousal' else pd.merge(df_existed, df)
    df.to_csv(pred_file, index=False)


def plot_video_prediction(df_pred, label_raw, partition, vid, emo_dim, save_dir):
    TIME_COLUMN = 'timestamp'
    EMO_COLUMN = 'value'

    df_pred = df_pred[df_pred['segment_id'] > 0]  # remove padding

    time = df_pred[TIME_COLUMN].values / 1000.0  # ms --> s
    pred = df_pred[EMO_COLUMN].values

    label_raw = [item for sublist in label_raw for item in sublist]
    label_raw = label_raw[:len(time)]

    label_target = savgol_filter(label_raw, 11, 3).tolist()

    # plot
    plt.figure(figsize=(20, 10))
    # color = 'r' if emo_dim == 'arousal' else 'g'
    plt.plot(time, pred, 'r', label=f'{emo_dim}(pred)')

    if label_raw is not None:
        plt.plot(time, label_raw, 'g', label=f'{emo_dim}(gt)')
    plt.plot(time, label_target, 'b', label=f'{emo_dim}(target)')

    plt.title(f"{emo_dim} of Video '{vid}' [{partition}]")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    # set margin on x axis
    ax = plt.gca()
    if time[-1] < 400:
        x_interval = 10
    elif time[-1] < 800:
        x_interval = 20
    else:
        x_interval = 50
    x_major_locator = plt.MultipleLocator(x_interval)
    ax.xaxis.set_major_locator(x_major_locator)
    # y_major_locator = plt.MultipleLocator(0.2)
    # ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim([-1, 1])
    plt.grid()

    plt.savefig(os.path.join(save_dir, f'{vid}.jpg'))
    plt.close()


def write_fusion_result(metas, preds, labels, params, partition, fusion, view=False):
    """
    :param metas: # video id, time stamp, segment id
    :param preds:
    :param params:
    :param partition:
    :param view: whether plot predicted arousal and valence or not
    :return:
    """
    # write prediction sample by sample (multiple files)
    if params.model == 'rnn':
        dir_name = f'{os.path.splitext(params.log_file_name)[0]}_[{params.n_seeds}_{params.current_seed}]_{fusion}'
    else:  # machine learning model
        dir_name = f'{os.path.splitext(params.log_file_name)[0]}_{fusion}_' + \
                   f'{params.label_preproc}_{params.savgol_window if params.label_preproc == "savgol" else ""}'
    csv_dir = os.path.join(params.base_dir, 'result', dir_name, 'csv')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    if view:
        img_dir = os.path.join(params.base_dir, 'result', dir_name, 'img')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

    for idx, emo_dim in enumerate(params.emo_dim_set):
        emo_dim_dir = os.path.join(csv_dir, emo_dim)
        if not os.path.exists(emo_dim_dir):
            os.mkdir(emo_dim_dir)
        columns = ['timestamp', 'value', 'segment_id']
        for meta, pred, label in zip(metas, preds, labels):
            vid = meta[0, 0]
            # csv
            sample_file_name = f'{vid}.csv'  # [vid].csv, ex: 1.csv
            sample_data = np.column_stack([meta[:, 1], pred[:, idx], meta[:, 2]])
            df = pd.DataFrame(sample_data, columns=columns)
            df[['timestamp', 'segment_id']] = df[['timestamp', 'segment_id']].astype(np.int)
            sample_pred_file = os.path.join(emo_dim_dir, sample_file_name)
            df.to_csv(sample_pred_file, index=False)

            # plot img
            if view:
                save_dir = os.path.join(img_dir, emo_dim)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                plot_video_prediction(df, label, partition, vid, emo_dim, save_dir)

    # write aggregated prediction (all in one file)
    metas = np.row_stack(metas)
    metas = metas[:, :2]
    preds = np.row_stack(preds)
    data = np.column_stack([metas, preds])
    columns = ['id', 'timestamp'] + ['prediction_' + emo_dim for emo_dim in params.emo_dim_set]
    df = pd.DataFrame(data, columns=columns)
    df[['id', 'timestamp']] = df[['id', 'timestamp']].astype(np.int)
    pred_file_name = f'{partition}.csv'
    aggr_dir = os.path.join(csv_dir, 'aggregated')
    if not os.path.exists(aggr_dir):
        os.mkdir(aggr_dir)
    pred_file = os.path.join(aggr_dir, pred_file_name)
    df.to_csv(pred_file, index=False)


def compute_EWE(annos):
    # annos:   list for each sequence with numpy arrays (num_annotators, seq_len, 1)
    # returns: list for each sequence with numpy arrays (seq_len, 1)

    num_annos = annos[0].shape[0]
    EWE, interrater = [], []

    for seq in annos:
        # Compute a weight for each annotation
        r = np.ones((num_annos, num_annos))

        # Note: The ones on the main diagonal should be kept
        for anno in range(num_annos):
            for anno_comp in range(num_annos):
                if anno != anno_comp:
                    r[anno, anno_comp], _, _ = cal_eval_metrics(seq[anno, :, 0], seq[anno_comp, :, 0])

        r_mean = np.zeros_like(r)
        r_mean = np.mean(r_mean, axis=1)
        for anno_0 in range(num_annos):
            for anno_1 in range(num_annos):
                if anno_0 != anno_1:
                    r_mean[anno_0] += r[anno_0, anno_1]
            r_mean[anno_0] /= num_annos - 1

        inter_r = np.nanmean(r, axis=1)
        inter_rater_agreement = np.mean(inter_r)

        r = r_mean

        r[np.isnan(r)] = 0.

        r[r < 0] = 0.  # Important: Give all negatively correlated annotations zero weight!

        r_sum = np.nansum(r)
        r = r / r_sum

        # Apply weights to get the Evaluator Weighted Estimator
        seq_len = seq.shape[1]
        EWEseq = np.zeros(seq_len)

        for anno in range(num_annos):
            EWEseq = np.round(EWEseq + seq[anno, :, 0] * r[anno], 3)

        # Normalised [-1,1]
        # EWEseq_norm = np.round(2. * (EWEseq - np.min(EWEseq)) / np.ptp(EWEseq) - 1, 4)

        EWE.append(EWEseq)
        interrater.append(inter_rater_agreement)

    return EWE
