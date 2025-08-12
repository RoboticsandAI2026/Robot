import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

class ObjectDetector:
    def __init__(self, model_path='final_object_classifier.keras'):
        self.img_height = 150
        self.img_width = 150
        self.model = tf.keras.models.load_model(model_path)
        self.class_labels = {
            0: 'cap',
            1: 'dust_pan',
            2: 'football',
            3: 'shoe',
            4: 'water_bottle'
        }

    def preprocess(self, img_array):
        img_array = tf.image.resize(img_array, [self.img_height, self.img_width])
        img_array = img_array / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        return img_array

    def detect(self, img_array):
        processed = self.preprocess(img_array)
        preds = self.model.predict(processed, verbose=0)[0]
        predicted_class_idx = np.argmax(preds)
        confidence = preds[predicted_class_idx] * 100
        predicted_label = self.class_labels[predicted_class_idx]
        return predicted_label, confidence
