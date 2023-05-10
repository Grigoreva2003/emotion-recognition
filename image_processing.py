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


# метрика оценки качества модели
def f1_score(y_true, y_pred):
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


# очищение директорию path от всех файлов расширения .jpg
def clean_folder(path):
    filelist = [f for f in os.listdir(path) if f.lower().endswith(".jpg")]
    for file in filelist:
        os.remove(os.path.join(path, file))


# получение эмоции на фотографии
def get_emotion():
    # extracting image for processing from the folder and predicting emotions
    test_dataset = test_datagen.flow_from_directory(directory='static/img/IO_img',
                                                    target_size=(48, 48),
                                                    class_mode='categorical',
                                                    batch_size=64,
                                                    shuffle=False)
    return fer_model.predict(test_dataset)[0]


# получение пола на фотографии
def get_gender():
    test_dataset = test_datagen.flow_from_directory(directory='static/img/IO_img',
                                                    target_size=(48, 48),
                                                    class_mode='binary',
                                                    batch_size=64,
                                                    shuffle=False)
    return gender_model.predict(test_dataset)[0][0]


def process_image(path_to_dir, filename):
    face_cascade = cv2.CascadeClassifier(
        'C:\Program Files (x86)\Python38-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

    img = cv2.imread(os.path.join(path_to_dir, filename))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotion_percentage = {val: 0 for val in emotion_dict.values()}
    gender_type = ''
    for (x, y, w, h) in faces:
        # resize cropped face from input image and convert it from GRAY to BGR
        resized_face = cv2.cvtColor(cv2.resize(crop_face(gray, x, y, w, h), (48, 48)), cv2.COLOR_GRAY2BGR)
        # saving image to "static/img/IO_img/prediction_img" folder
        cv2.imwrite(os.path.join(path_to_dir + '\prediction_img', 'cropped_face.jpg'), resized_face)

        emotion_prediction = get_emotion()
        gender_prediction = get_gender()

        gender_type = gender_dict[gender_prediction > 0.5]
        emotion_type = emotion_dict[np.argmax(emotion_prediction)]

        # заполняем словарь распределения эмоций для отображения на круговой диаграмме
        for (i, name) in zip(range(7), emotion_dict.values()):
            emotion_percentage[name] += emotion_prediction[i]

        draw_info(img, x, y, w, h, emotion_type)

    if len(faces) > 0:
        emotion_percentage = [round(emotion_percentage[key] * 100 / len(faces)) for key in emotion_percentage.keys()]
        emotion_percentage[-1] = 100 - sum(emotion_percentage[:-1])
    if len(faces) > 1:
        gender_type = 'для предсказания необходимо наличие единственного лица на изображении'

    clean_folder(path_to_dir)

    output_filename = 'output_' + filename
    cv2.imwrite(os.path.join(path_to_dir, output_filename), img)
    return output_filename, emotion_percentage, gender_type


emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
gender_dict = {0: 'женщина', 1: 'мужчина'}

fer_model = keras.models.load_model('fer_model_tf.h5',
                                    custom_objects={'f1_score': f1_score},
                                    compile=False)
gender_model = keras.models.load_model('gender_model_tf.h5',
                                       compile=False)

test_datagen = ImageDataGenerator(rescale=1. / 255)
