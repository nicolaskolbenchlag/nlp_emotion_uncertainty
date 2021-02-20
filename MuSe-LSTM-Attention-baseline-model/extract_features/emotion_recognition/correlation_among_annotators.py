import torch
import utils
from dataset import MyDatasetStdv

import pandas as pd

def evaluate(model, test_loader, params):
    model.eval()
    full_preds, full_labels = [], []
    with torch.no_grad():
        for batch, batch_data in enumerate(test_loader, 1):
            features, feature_lens, labels, meta, stdvs = batch_data
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

def get_dataloaders(params, data, stdvs, predict=False):
    sample_counts = []
    data_loader = {}
    for partition in data.keys():
        set_ = MyDatasetStdv(data, partition, stdvs)
        sample_counts.append(len(set_))

        batch_size = params.batch_size if partition == 'train' and not predict else 1
        shuffle = True if partition == 'train' else False
        data_loader[partition] = torch.utils.data.DataLoader(set_, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        
    print(f'Samples in partitions: {*sample_counts,}')
    return data_loader


def get_correlations_data_loader(params):
    # NOTE: calculate correlation among annotators
    all_annotations = {}
    annotators = [2, 4, 7, 5, 8]#range(1, 16)
    for annotator in annotators:
        data = utils.load_data(params, params.feature_set, params.emo_dim_set, params.normalize, params.label_preproc, params.norm_opts, params.segment_type, params.win_len, params.hop_len, save=params.cache, refresh=params.refresh, add_seg_id=params.add_seg_id, annotator=annotator)
        
        for partition in ["devel", "test", "train"]:
            metas = data[partition]["meta"]
            labels = data[partition]["label"]
            
            labels = [[ts[0] for ts in sample] for sample in labels]# NOTE: gets first predicted emo_dim / only works with 1
            
            for i, meta in enumerate(metas):

                vid_id = meta[0][0]
                timestamps = meta[1]
                seqgment_ids = meta[2]
                
                label = labels[i]

                if vid_id not in all_annotations.keys():
                    all_annotations[vid_id] = []
                
                all_annotations[vid_id].append(label)
    
    ccs = {}
    for vid_id, annotations in all_annotations.items():
        ccs_tmp = []

        for annotation_1 in annotations:
            for annotation_2 in annotations:
                if annotation_1 != annotation_2:
                    cc = pd.Series(annotation_1).corr(pd.Series(annotation_2))
                    ccs_tmp.append(cc)

        mean = torch.tensor(ccs_tmp).float().mean().abs()
        mean = (mean + 1) / 2 if not torch.isnan(mean).item() else torch.tensor(1.)
        ccs[vid_id] = mean


    data = utils.load_data(params, params.feature_set, params.emo_dim_set, params.normalize, params.label_preproc, params.norm_opts, params.segment_type, params.win_len, params.hop_len, save=params.cache, refresh=params.refresh, add_seg_id=params.add_seg_id, annotator=None)
    data_loader = get_dataloaders(params, data, ccs)

    data_loader_gt = None

    return data_loader, data_loader_gt, data