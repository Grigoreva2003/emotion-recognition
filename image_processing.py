import cv2, os

import tensorflow as tf
import numpy as np
import keras.backend as K

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# text font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
font_scale = 1
# Element's color
line_color = (197, 48, 63)
text_color = (255, 255, 255)
# Line thickness of 2 px
thickness = 2


def f1_score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def crop_face(img, x, y, w, h):
    return img[y:(y + h), x:(x + w)]


def draw_info(img, x, y, w, h, emotion_text):
    # rectangle frame for face
    cv2.rectangle(img, (x, y), (x + w, y + h), line_color, thickness)
    # getting size of predicted emotion text
    text_size, _ = cv2.getTextSize(emotion_text, font, font_scale, thickness)
    text_w, text_h = text_size
    # drawing rectangle for the text inside frame
    cv2.rectangle(img, (x, y), (x + text_w, y + text_h + 5), line_color, -1)
    # putting text inside rectangle
    cv2.putText(img, emotion_text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, thickness,
                cv2.LINE_AA)


def process_image(path_to_dir, filename):
    face_cascade = cv2.CascadeClassifier(
        'C:\Program Files (x86)\Python38-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

    img = cv2.imread(os.path.join(path_to_dir, filename))

    filelist = [f for f in os.listdir(path_to_dir) if f.endswith(".jpg")]
    for file in filelist:
        os.remove(os.path.join(path_to_dir, file))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # resize cropped face from input image and convert it from GRAY to BGR
        resized_face = cv2.cvtColor(cv2.resize(crop_face(gray, x, y, w, h), (48, 48)), cv2.COLOR_GRAY2BGR)
        # saving image to "static/img/IO_img/prediction_img" folder
        cv2.imwrite(os.path.join(path_to_dir + '\prediction_img', 'test.jpg'), resized_face)

        # extracting image for processing from the folder and predicting emotions
        test_dataset = test_datagen.flow_from_directory(directory='static/img/IO_img',
                                                        target_size=(48, 48),
                                                        class_mode='categorical',
                                                        batch_size=64,
                                                        shuffle=False)
        prediction = model.predict(test_dataset)[0]
        emotion_type = emotion_dict[np.argmax(prediction)]

        draw_info(img, x, y, w, h, emotion_type)

    output_filename = 'output_' + filename
    cv2.imwrite(os.path.join(path_to_dir, output_filename), img)
    return output_filename


emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

model = keras.models.load_model('fer_model_tf.h5',
                                custom_objects={'f1_score': f1_score},
                                compile=False)

test_datagen = ImageDataGenerator(rescale=1. / 255)
# test_dataset = test_datagen.flow_from_directory(directory='static/img',
#                                                 target_size=(48, 48),
#                                                 class_mode='categorical',
#                                                 batch_size=64,
#                                                 shuffle=False)
# res = model.predict(test_dataset)
# for ans in res:
#     print('Answer for the 0 photo:', emotion_dict[np.argmax(ans)])
