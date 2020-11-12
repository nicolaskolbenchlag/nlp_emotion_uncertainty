# *_*coding:utf-8 *_*
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import vggish_input
import vggish_params
import vggish_slim

import pandas as pd
import glob
import config

MAX_BATCH_SIZE = 3000


def extract_acoustic_feature(wav_file):
    samples = vggish_input.wavfile_to_examples(wav_file)  # sample_size * height(96) * width(64)
    sample_size = samples.shape[0]
    print(f'Sample size: {sample_size}')
    # max sample size: 6653, will cause OOM. Need to chunk samples.
    if sample_size > MAX_BATCH_SIZE:
        examples_batches = [samples[:MAX_BATCH_SIZE], samples[MAX_BATCH_SIZE:]]
    else:
        examples_batches = [samples]

    embeddings = []
    for examples_batch in examples_batches:
        with tf.Graph().as_default(), tf.Session() as sess:
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt')
            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

            # Run inference and postprocessing.
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: examples_batch})
            embeddings.append(embedding_batch)

    embeddings = np.row_stack(embeddings)
    print(f'Embedding shape: {embeddings.shape}.')

    return embeddings


# problem: OOM
def extract_acoustic_feature_org(wav_file):
    examples_batch = vggish_input.wavfile_to_examples(wav_file)  # sample_size * height(96) * width(64)
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt')
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})

        print(f'Embedding shape: {embedding_batch.shape}.')

        return embedding_batch


def write_feature_to_csv(features, out_file):
    n_frames, feature_dim = features.shape
    hop_len = int(vggish_params.EXAMPLE_HOP_SECONDS * 1000)
    timestamps = np.arange(n_frames) * hop_len + 500  # 500 ~= win_len / 2 (i.e. 0.975*1000/2)
    first_timestamp, last_timestamp = timestamps[0], timestamps[-1]

    data = np.column_stack([timestamps, features])
    meta_columns = ['timestamp', 'segment_id']
    timstamp_column = meta_columns[0]
    columns = [timstamp_column] + [str(i) for i in range(feature_dim)]
    df = pd.DataFrame(data, columns=columns)
    df[timstamp_column] = df[timstamp_column].astype(np.int64)

    out_file_name = os.path.basename(out_file)
    label_file = os.path.join(config.PATH_TO_LABELS, 'arousal', out_file_name)
    if not os.path.exists(label_file):
        return
    df_label = pd.read_csv(label_file)
    metas = df_label[meta_columns].values
    label_first_timestamp, label_last_timestamp = metas[0, 0], metas[-1, 0]

    # pad in head if necessary
    if first_timestamp > label_first_timestamp:
        n_pad_frames = int((first_timestamp - label_first_timestamp) / hop_len)
        pad_timestamps = np.arange(label_first_timestamp, first_timestamp, hop_len)
        print(f'Note: label first timestamp ({label_first_timestamp}) < feature first timestamp ({first_timestamp}). '
              f'Pad first frame (<--) of feature for timestamp: {pad_timestamps.tolist()}.')
        pad_features = np.tile(df.iloc[0].values[1:], (n_pad_frames, 1))
        pad_data = np.column_stack([pad_timestamps, pad_features])
        data = np.row_stack([pad_data, data])
    # pad in tail if necessary
    if last_timestamp < label_last_timestamp:
        n_pad_frames = int((label_last_timestamp - last_timestamp) / hop_len)
        pad_timestamps = np.arange(last_timestamp, label_last_timestamp, hop_len) + hop_len
        print(f'Note: feature last timestamp ({last_timestamp}) < label last timestamp ({label_last_timestamp}). '
              f'Pad last frame (-->) of feature for timestamp: {pad_timestamps.tolist()}.')
        pad_features = np.tile(df.iloc[-1].values[1:], (n_pad_frames, 1))
        pad_data = np.column_stack([pad_timestamps, pad_features])
        data = np.row_stack([data, pad_data])
    df = pd.DataFrame(data, columns=columns)
    df[timstamp_column] = df[timstamp_column].astype(np.int64)
    first_timestamp, last_timestamp = df.iloc[0, 0], df.iloc[-1, 0]
    assert first_timestamp <= label_first_timestamp and last_timestamp >= label_last_timestamp, 'Error!'

    label_aligned_features = df[df[timstamp_column].isin(df_label[timstamp_column])].values[:, 1:]
    columns = meta_columns + [str(i) for i in range(feature_dim)]
    data = np.column_stack([metas, label_aligned_features])
    df = pd.DataFrame(data, columns=columns)
    df[meta_columns] = df[meta_columns].astype(np.int64)
    df.to_csv(out_file, index=False)


def write_feature_to_csv_org(features, out_file):
    n_frames, feature_dim = features.shape
    hop_len = int(vggish_params.EXAMPLE_HOP_SECONDS * 1000)
    timestamps = np.arange(1, n_frames + 1) * hop_len  # TODO: np.arange(n_frames) * hop_len + win_len / 2
    last_timestamp = timestamps[-1]

    data = np.column_stack([timestamps, features])
    meta_columns = ['timestamp', 'segment_id']
    timstamp_column = meta_columns[0]
    columns = [timstamp_column] + [str(i) for i in range(feature_dim)]
    df = pd.DataFrame(data, columns=columns)
    df[timstamp_column] = df[timstamp_column].astype(np.int64)

    out_file_name = os.path.basename(out_file)
    label_file = os.path.join(config.PATH_TO_LABELS, 'arousal', out_file_name)
    df_label = pd.read_csv(label_file)
    metas = df_label[meta_columns].values
    label_last_timestamp = metas[-1, 0]

    if last_timestamp < label_last_timestamp:
        n_pad_frames = int((label_last_timestamp - last_timestamp) / hop_len)
        pad_timestamps = np.arange(last_timestamp, label_last_timestamp, hop_len) + hop_len
        print(f'Note: feature last timestamp ({last_timestamp}) < label last timestamp ({label_last_timestamp}). '
              f'Pad last frame of feature for timestamp: {pad_timestamps.tolist()}.')
        pad_features = np.tile(df.iloc[-1].values[1:], (n_pad_frames, 1))
        pad_data = np.column_stack([pad_timestamps, pad_features])
        data = np.row_stack([data, pad_data])
        df = pd.DataFrame(data, columns=columns)
        df[timstamp_column] = df[timstamp_column].astype(np.int64)
        last_timestamp = df.iloc[-1, 0]
        assert last_timestamp == label_last_timestamp, 'Error!'

    label_aligned_features = df[df[timstamp_column].isin(df_label[timstamp_column])].values[:, 1:]
    columns = meta_columns + [str(i) for i in range(feature_dim)]
    data = np.column_stack([metas, label_aligned_features])
    df = pd.DataFrame(data, columns=columns)
    df[meta_columns] = df[meta_columns].astype(np.int64)
    df.to_csv(out_file, index=False)


# Note: almost close between org and new
def test():
    # in & out
    audio_path = config.PATH_TO_RAW_AUDIO
    wav_file = os.path.join(audio_path, '248.wav')

    print(f'Using org...')
    features_org = extract_acoustic_feature_org(wav_file)
    print(f'Using new...')
    features = extract_acoustic_feature(wav_file)
    assert np.equal(features, features_org).all()
    print('Seems good to me!')


def main():
    # in & out
    audio_path = config.PATH_TO_RAW_AUDIO
    out_dir = 'vggish'  # os.path.join(config.PATH_TO_ALIGNED_FEATURES, 'vggish2')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    wav_files = glob.glob(os.path.join(audio_path, '*.wav'))
    n_tasks = len(wav_files)
    for i, wav_file in enumerate(wav_files, 1):
        print(f'{i}/{n_tasks}: processing {os.path.basename(wav_file)} ...')
        features = extract_acoustic_feature(wav_file)

        out_file_name = os.path.basename(wav_file).split('.')[0] + '.csv'
        out_file = os.path.join(out_dir, out_file_name)
        write_feature_to_csv(features, out_file)


if __name__ == '__main__':
    main()
    # test()
