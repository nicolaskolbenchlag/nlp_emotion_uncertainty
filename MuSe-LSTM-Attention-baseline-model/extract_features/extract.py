import os
import glob
import pandas as pd
import re
import argparse
import numpy as np


def parse_timestamps_from_labels(file):
    df = pd.read_csv(file)
    return df[['timestamp', 'segment_id']]


def align_with_timestamps(timestamps, segment_ids, df, name):
    """
    Sum of values that fall into range(timestamp, timestamp + 1).
    If there is no value for this timestamp, use data of previous timestamp.
    If there is no previous timestamp, use 0.0.
    """
    rows = []
    for time, seg_id in zip(timestamps, segment_ids):
        row = df[df['start'].between(time, time + 250)]
        values = list(row.iloc[:, 3:].sum().values)
        if row.empty:
            if len(rows) == 0:
                values = [0.0] * feature_dims[name]['feature_count']
            else:
                values = list(rows[-1].values[0][2:])
        col_names = ['timestamp'] + df.columns.tolist()[2:]
        row = [time, seg_id] + values
        row = pd.DataFrame([row], columns=col_names)
        rows.append(row)

    aligned_df = pd.concat(rows, ignore_index=True)
    return aligned_df


def add_timestamps(label_df, data_df):
    masked_data = data_df[data_df['timestamp'].isin(label_df['timestamp'].tolist())]

    mask = np.isin(label_df['timestamp'].values, masked_data['timestamp'].values)
    missing_timestamps = pd.DataFrame(label_df[['timestamp', 'segment_id']].values[~mask],
                                      columns=pd.Index(['timestamp', 'segment_id']))
    format_df = masked_data.append(missing_timestamps, ignore_index=True, sort=False)
    format_df = format_df.sort_values(by=['timestamp'])
    format_df.index = pd.RangeIndex(len(format_df.index))
    format_df.index = range(len(format_df.index))
    format_df = format_df.fillna(0)
    return format_df


def extract_text_features(name, out_path):
    from extract_text_features import extract_bert_features
    transcription_path = os.path.join(data_feat_path, 'transcription_segments')
    vid_ids = [int(i) for i in os.listdir(transcription_path)]

    for vid_id in vid_ids:
        if not os.listdir(os.path.join(out_path, str(vid_id) + '.csv')):
            print('Video ', str(vid_id))

            files = glob.glob(os.path.join(transcription_path, str(vid_id)) + '/*.csv')
            files.sort(key=lambda f: int(re.sub('\D', '', f)))

            data = []
            for file in files:
                df = pd.read_csv(file, delimiter=',')
                words = df['word'].tolist()
                words = [txt.replace(' - ', ' ') for txt in words]

                df = df.drop(['word'], axis=1)
                if name in ['bert', 'albert']:
                    df = extract_bert_features(df, words, file, feature_dims[name]['num_hidden_layers'],
                                               feature_dims[name]['feature_count'], name)
                data.append(df)

            data_df = pd.concat(data, ignore_index=True)
            print('\tExtracted {} Features.'.format(name))

            label_df = parse_timestamps_from_labels(os.path.join(label_path, str(vid_id) + '.csv'))
            timestamps = label_df['timestamp'].tolist()
            segment_ids = label_df['segment_id'].tolist()
            final_df = align_with_timestamps(timestamps, segment_ids, data_df, name)
            print('\tAligned with timestamps.')

            final_df.to_csv(os.path.join(out_path, str(vid_id) + '.csv'), index=False)
            print('\tSaved to file.')


def extract_visual_features(name, out_path):
    from extract_visual_features import extract_faces, extract_vgg2_features
    videos_path = os.path.join(data_raw_path, 'video/')

    faces_path = 'faces/'  # Extract faces if they haven't been
    if not os.path.exists(faces_path):
        os.makedirs(faces_path)
        video_paths = [os.path.join(videos_path, dir) for dir in os.listdir(videos_path)]
        for path in video_paths:
            vid_id = path.split('/')[-1].split('_')[0]
            dst_folder = os.path.join(faces_path, vid_id) + '/'
            print('Extracting faces for video {}'.format(vid_id))
            os.mkdir(dst_folder)

            extract_faces(path, 4, dst_folder)  # 4 frames per second

            num_faces = len(os.listdir(dst_folder))
            print('\tFound {} faces.'.format(num_faces))

    for vid_id in os.listdir(faces_path):
        if not os.listdir(os.path.join(out_path, str(vid_id) + '.csv')):
            print('Video ', str(vid_id))
            if name == 'vgg2':
                data_df = extract_vgg2_features(os.path.join(faces_path, vid_id))
            label_df = parse_timestamps_from_labels(os.path.join(label_path, vid_id + '.csv'))
            final_df = add_timestamps(label_df, data_df)
            print('\tAligned with timestamps.')

            final_df.to_csv(os.path.join(out_path, str(vid_id) + '.csv'), index=False)
            print('\tSaved to file.')
        else:
            print('Features found for video {}.'.format(vid_id))


def extract_audio_features(name, out_path):
    from extract_audio_features import extract_vggish_features
    audio_path = os.path.join(data_feat_path, 'audio_segments')
    for vid_id in os.listdir(audio_path):
        if not os.listdir(os.path.join(out_path, str(vid_id) + '.csv')):
            print('Extracting {} features for video {}:'.format(name, vid_id))

            label_df = parse_timestamps_from_labels(os.path.join(label_path, vid_id + '.csv'))
            data_df = extract_vggish_features(os.path.join(audio_path, str(vid_id)), label_df)

            print('\tExtracted {} Features.'.format(name))

            final_df = add_timestamps(label_df, data_df)
            print('\tAligned with timestamps.')

            final_df.to_csv(os.path.join(out_path, str(vid_id) + '.csv'), index=False)
            print('\tSaved to file.')


if __name__ == '__main__':
    #### Arg Parser ####
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--data_path', dest='data_path', required=False, action='store',
                        default='C:/Users/Lea/PycharmProjects/MuSe-annotations/data/',
                        help='Specify path to data folder.')
    parser.add_argument('-n', '--feature_name', dest='feature_name', required=False, action='store',
                        default='vgg2',
                        help='Specify name of features that are to be extracted.')

    args = parser.parse_args()

    #### Paths ####

    data_path = args.data_path
    data_raw_path = os.path.join(data_path, 'raw/')
    data_feat_path = os.path.join(data_path, 'muse-wild/')
    label_path = os.path.join(data_feat_path, 'label_segments/raw_annotations/arousal/')

    #### Features ####

    feature_types = {'vgg2': 'visual', 'bert': 'text', 'albert': 'text', 'vggish': 'audio'}
    feature_dims = {'bert': {'num_hidden_layers': 2, 'feature_count': 2 * 768},
                    'albert': {'num_hidden_layers': 2, 'feature_count': 2 * 768}, 'vgg2': {'feature_count': 2048},
                    'vggish': {'feature_count': 1}}

    feature_name = args.feature_name

    #### Extraction ####

    out_path = feature_name + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if feature_types[feature_name] == 'text':
        extract_text_features(feature_name, out_path)
    elif feature_types[feature_name] == 'visual':
        extract_visual_features(feature_name, out_path)
    elif feature_types[feature_name] == 'audio':
        extract_audio_features(feature_name, out_path)
