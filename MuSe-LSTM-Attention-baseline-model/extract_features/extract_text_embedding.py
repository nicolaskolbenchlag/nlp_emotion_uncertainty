# *_*coding:utf-8 *_*
import os
import glob
import math
import pandas as pd
import numpy as np
import torch
import time
from tqdm import tqdm
import itertools
from transformers import AutoModel, AutoTokenizer

import config

LOG_DIR = 'output/log'
BERT_BASE = 'bert-base-uncased'
BERT_LARGE = 'bert-large-uncased'


def extract_bert_embedding(model_name=BERT_BASE, layer_ids=None, combine_type='mean',
                           batch_size=256, gpu=6, dir_name=None, trans_path=config.PATH_TO_TRANSCRIPTIONS):
    """
    :param model_name: which pretrained model
    :param layer_ids: output of selected layers will be summed as the token embedding. "-1" denotes output of last layer
    :param combine_type: how to combine subword embeddings to construct the whole word embedding.
    :param batch_size:
    :param gpu:
    :param trans_path: path to transcription
    :return:
    """
    start_time = time.time()
    # dir
    type = model_name.split('-')[0]
    if dir_name is None:
        dir = os.path.join(config.PATH_TO_ALIGNED_FEATURES, type)
    else:
        dir = dir_name  # os.path.join(config.PATH_TO_ALIGNED_FEATURES, dir_name)
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        print('Warning: out dir already exists!')
    # layer ids
    if layer_ids is None:
        layer_ids = [-4, -3, -2, -1]
    else:
        assert isinstance(layer_ids, list)
    # load model and tokenizer
    print('Loading pretrained tokenizer and model...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # device = torch.device(f'cuda:{gpu}')
    model.cuda()
    model.eval()
    # iterate videos
    vids = sorted(os.listdir(trans_path), key=lambda x: int(x))
    assert len(vids) == 291
    for vid in tqdm(vids):
        # for vid in vids:
        print(f' Processing {vid}...')
        # extract sentences and timestamps from raw files
        v_dir = os.path.join(trans_path, vid)
        trans_files = glob.glob(os.path.join(v_dir, '*.csv'))
        trans_files.sort(key=lambda x: int(os.path.basename(os.path.splitext(x)[0]).split('_')[1]))
        segment_dfs = []
        for file in trans_files:
            segment_df = pd.read_csv(file)
            segment_dfs.append(segment_df)
        df = pd.concat(segment_dfs)
        sentences, sentence = [], []
        timestamps = []
        for _, row in df.iterrows():
            word = row['word']
            if word in ['.', '!', '?']:
                # assert sentence != [], print(timestamps[-1])
                if sentence != []:
                    sentences.append(sentence)
                    sentence = []
            else:
                s_t, e_t = row['start'], row['end']
                if (e_t - s_t) > 1:  # denotes word (for punctuation, the interval is always 1)
                    sentence.append(word.lower())  # to lower case
                    timestamps.append((s_t, e_t))
        if sentence != []:  # some files do not end with '.', '!', '?'
            sentences.append(sentence)
        words = list(itertools.chain(*sentences))
        assert len(words) == len(timestamps), print(sentence)
        # extract embedding
        embeddings = []
        n_batches = math.ceil(len(sentences) / batch_size)
        for i in range(n_batches):
            s_idx, e_idx = i * batch_size, min((i + 1) * batch_size, len(sentences))
            batch_sentences = sentences[s_idx:e_idx]
            inputs = tokenizer(batch_sentences, padding=True, is_pretokenized=True, return_tensors='pt')
            input_ids = inputs['input_ids']
            #print(inputs)
            inputs['attention_mask'] = inputs['attention_mask'].cuda()
            inputs['input_ids'] = inputs['input_ids'].cuda()
            inputs['token_type_ids'] = inputs['token_type_ids'].cuda()
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)[2]
                outputs = torch.stack(outputs)[layer_ids].sum(dim=0)  # sum
                outputs = outputs.cpu().numpy()  # (B, T, D)
                lens = torch.sum(inputs['attention_mask'], dim=1)
                real_batch_size = outputs.shape[0]
                for i in range(real_batch_size):
                    input_id = input_ids[i, 1:(lens[i] - 1)]  # (T,) Note, 1:(lens[i] - 1) for skipping [CLS] and [SEP]
                    output = outputs[i, 1:(lens[i] - 1)]  # (T, D)
                    sentence = batch_sentences[i]
                    n_tokens, n_words = output.shape[0], len(sentence)
                    if n_tokens == n_words:  # sub-word is word
                        sentence_embedding = list(output)
                        embeddings.extend(sentence_embedding)
                    else:  # align sub-word to word
                        sentence_embedding = []
                        pointer = 0
                        word, word_embedding = '', []
                        for j, token_id in enumerate(input_id):
                            token = tokenizer.convert_ids_to_tokens([token_id])[0]
                            token_embedding = output[j]
                            current_word = sentence[pointer]
                            token = token.replace('▁', '')  # for albert model (ex, hello: '▁hello')
                            if token == current_word:
                                sentence_embedding.append(token_embedding)
                                pointer += 1
                            else:
                                word_embedding.append(token_embedding)
                                token = token.replace('#', '')  # for bert model
                                word = word + token
                                if word == current_word:  # ended token
                                    if combine_type == 'sum':
                                        word_embedding = np.sum(np.row_stack(word_embedding),
                                                                axis=0)  # sum sub-word emebddings
                                    elif combine_type == 'mean':
                                        word_embedding = np.mean(np.row_stack(word_embedding),
                                                                 axis=0)  # average sub-word emebddings
                                    elif combine_type == 'last':
                                        word_embedding = word_embedding[-1]  # take the last sub-word emebdding
                                    else:
                                        raise Exception('Error: not supported type to combine subword embedding.')
                                    sentence_embedding.append(word_embedding)
                                    word, word_embedding = '', []
                                    pointer += 1
                        assert len(sentence) == len(sentence_embedding), print(sentence)
                        embeddings.extend(sentence_embedding)
        assert len(embeddings) == len(timestamps)
        # align with label timestamp and write csv file
        csv_file = os.path.join(dir, f'{vid}.csv')
        log_file = os.path.join(LOG_DIR, type, f'{vid}.csv')
        if not os.path.exists(os.path.join(LOG_DIR, type)):
            os.makedirs(os.path.join(LOG_DIR, type))
        write_embedding_to_csv(embeddings, timestamps, words, csv_file, log_file=log_file)
    end_time = time.time()
    print(f'Time used (bert): {end_time - start_time:.1f}s.')


def write_embedding_to_csv(embeddings, timestamps, words, csv_file, log_file=None):
    # get label file
    vid = os.path.basename(os.path.splitext(csv_file)[0])
    label_file = os.path.join(config.PATH_TO_LABELS, 'arousal', f'{vid}.csv')
    df_label = pd.read_csv(label_file)
    meta_columns = ['timestamp', 'segment_id']
    metas = df_label[meta_columns].values
    label_timestamps = metas[:, 0]
    # align word, timestamp & embedding
    embedding_dim = len(embeddings[0])
    n_frames = len(label_timestamps)
    aligned_embeddings = np.zeros((n_frames, embedding_dim))
    aligned_timestamps = np.empty((n_frames, 2), dtype=np.object)
    aligned_words = np.empty((n_frames,), dtype=np.object)
    label_timestamp_idxs = np.arange(n_frames)
    hit_count = 0
    for i, (s_t, e_t) in enumerate(timestamps):
        idxs = label_timestamp_idxs[np.where((label_timestamps >= s_t) & (label_timestamps < e_t))]
        if len(idxs) > 0:
            aligned_embeddings[idxs] = embeddings[i]
            aligned_timestamps[idxs] = [int(s_t), int(e_t)]
            aligned_words[idxs] = words[i]
            hit_count += len(idxs)
    print(f'Video "{vid}" hit rate: {hit_count / n_frames:.1%}.')
    # write csv file
    columns = meta_columns + [str(i) for i in range(embedding_dim)]
    data = np.column_stack([metas, aligned_embeddings])
    df = pd.DataFrame(data=data, columns=columns)
    df[meta_columns] = df[meta_columns].astype(np.int64)
    df.to_csv(csv_file, index=False)
    # write log file
    if log_file is not None:
        log_columns = meta_columns + ['start', 'end', 'word']
        log_data = np.column_stack([metas, aligned_timestamps, aligned_words])
        log_df = pd.DataFrame(data=log_data, columns=log_columns)
        log_df[meta_columns] = log_df[meta_columns].astype(np.int64)
        log_df.to_csv(log_file, index=False)
    return data


if __name__ == '__main__':
    # extract_bert_embedding(BERT_BASE, layer_ids=[-1], dir_name='bert', gpu=3)
    # extract_bert_embedding(BERT_BASE, layer_ids=[-2, -1], dir_name='bert-2', gpu=3)
    extract_bert_embedding(BERT_BASE, layer_ids=[-4, -3, -2, -1], dir_name='bert-4', gpu=3)
