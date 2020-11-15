# *_*coding:utf-8 *_*
import os

# path
# PATH_TO_MUSE_2020 = 'C:/Users/Lea/PycharmProjects/MuSe-annotations/data/'# change this path to yours  #'../../../databases/MuSe-data-base/data/' #

# PATH_TO_ALIGNED_FEATURES = os.path.join(PATH_TO_MUSE_2020, 'muse-wild/feature_segments/label_aligned')
# PATH_TO_UNALIGNED_FEATURES = os.path.join(PATH_TO_MUSE_2020, 'muse-wild/feature_segments/unaligned')
# PATH_TO_LABELS = os.path.join(PATH_TO_MUSE_2020, 'muse-wild/label_segments')
# PATH_TO_LABELS_RAW = os.path.join(PATH_TO_LABELS, 'raw_annotations')
# PATH_TO_TRANSCRIPTIONS = os.path.join(PATH_TO_MUSE_2020, 'muse-wild/transcription_segments')

# PATH_TO_METEDATA = os.path.join(PATH_TO_MUSE_2020, 'raw/metadata')
# PARTITION_FILE = os.path.join(PATH_TO_METEDATA, 'partition.csv')
# META_FILE = os.path.join(PATH_TO_METEDATA, 'video_metadata.csv')
# ANNOTATOR_MAPPING = 'annotator_id_mapping.json'

# PATH_TO_RAW_AUDIO = os.path.join(PATH_TO_MUSE_2020, 'raw/audio_norm')
# PATH_TO_RAW_VIDEO = os.path.join(PATH_TO_MUSE_2020, 'raw/video')

DATA_FOLDER = 'MuSe-LSTM-Attention-baseline-model/output/data'
MODEL_FOLDER = 'MuSe-LSTM-Attention-baseline-model/output/model'
LOG_FOLDER = 'MuSe-LSTM-Attention-baseline-model/output/log'
PREDICTION_FOLDER = 'MuSe-LSTM-Attention-baseline-model/output/prediction'
# FUSION_FOLDER = 'output/fusion'

PARTITION_FILE = "c1_muse_wild/partition.csv"

PATH_TO_ALIGNED_FEATURES = "MuSe-LSTM-Attention-baseline-model/extracted_features/"

PATH_TO_LABELS = "c1_muse_wild/label_segments/"
# PATH_TO_LABELS_RAW = PATH_TO_MUSE_2020 + "raw/annotations/raw_annotations/"
PATH_TO_TRANSCRIPTIONS = "c1_muse_wild/transcription_segments/"

# numerical
EPSILON = 1e-6
