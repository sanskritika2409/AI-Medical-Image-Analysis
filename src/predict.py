import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/medical_model.h5")

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return "PNEUMONIA"
    else:
        return "NORMAL"