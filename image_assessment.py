import tensorflow as tf
from src.utils.utils import calc_mean_score
from src.handlers.model_builder import Nima
import numpy as np
from src.handlers.data_generator import TestDataGenerator
import os

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

class ImageAssessment:
    def __init__(self, weights_file):
        self.base_model_name = "MobileNet"
        self.weights_file = weights_file

        # Set GPU
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            with self.sess.as_default():
                self.nima = Nima(self.base_model_name, weights=None)
                self.nima.build()
                self.nima.nima_model.load_weights(self.weights_file)

    def assessment(self, img_arr, type="aesthetic"):
        X = np.array([img_arr])
        X = self.nima.base_module.preprocess_input(X)
        with self.graph.as_default():
            with self.sess.as_default():
                predictions = self.nima.nima_model.predict(X)
        new_score = self.count_score(predictions[0], type)
        return new_score

    def assessment_video(self, img_arr_list, type="aesthetic"):

        data_generator = TestDataGenerator(img_arr_list, 32, 10, self.nima.preprocessing_function())

        with self.graph.as_default():
            with self.sess.as_default():
                predictions = self.nima.nima_model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)
        new_scores = []
        for prediction in predictions:
            new_score = self.count_score(prediction, type)
            new_scores.append(new_score)
        return np.array(new_scores)

    def count_score(self, sc, type="aesthetic"):
        if type == "aesthetic":
            aesthetic_score = round(calc_mean_score(sc) * 10.0 / 8.0, 1) * 10
            return aesthetic_score
        else:
            technical_score = round(calc_mean_score(sc) * 10.0 / 7.5, 1) * 10
            return technical_score







