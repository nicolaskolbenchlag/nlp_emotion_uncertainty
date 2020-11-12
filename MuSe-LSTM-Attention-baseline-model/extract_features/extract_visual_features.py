# Adapted from https://github.com/lstappen/MuSe-data-base/blob/master/feature_extraction/vgg.py
import cv2
import dlib
import numpy as np
import os
import pandas as pd

from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


def crop_image(image, bounding_box):
    x1, y1, x2, y2 = bounding_box
    return image[y1:y2, x1:x2]


def extract_faces(video_path, frames_per_second, dest_path, padding=0.0):
    def save_cropped_image(image, face, path):
        rect = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
        x, y, width, height = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
        if padding != 0.0:
            x, y = x
        image = crop_image(image, (x, y, x + width, y + height))
        if image.size:
            cv2.imwrite(path, image)

    # Download pretrained face detector model from: http://dlib.net/files/mmod_human_face_detector.dat.bz2
    detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    multiplier = fps / frames_per_second
    next_sample_frame = 1
    ms = 0

    success, image = capture.read()
    while success:
        frame_id = int(round(capture.get(1)))
        if frame_id == int(next_sample_frame):
            next_sample_frame += multiplier
            faces = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if len(faces):
                for i, face in enumerate(faces):
                    if face.confidence > 0.6:  # Save faces only for confidence greater than 0.6
                        save_path = dest_path + "{}_{}.jpg".format(ms, i)
                        save_cropped_image(image, face, save_path)
            ms += int(1000 / frames_per_second)
        success, image = capture.read()
    capture.release()


def extract_vgg2_features(path, feature_count=2048):
    model = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3), pooling='avg')

    features = []
    for file in os.listdir(path):

        timestamp = file.split('_')[0]
        img = image.load_img(os.path.join(path, file), target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x, version=2)
        out = model.predict(x).tolist()[0]
        features.append([int(timestamp)] + out)

    cols = [str(x) for x in range(feature_count)]
    features = pd.DataFrame(features, columns=['timestamp'] + cols)
    features = features.sort_values(by='timestamp')
    return features
