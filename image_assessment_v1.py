
import os
import glob
import numpy as np
from src.utils import utils
from src.utils.utils import calc_mean_score, save_json
from src.handlers.model_builder import Nima
from src.handlers.data_generator import TestDataGenerator

class ImageAssessment:
    def __init__(self):
        self.base_model_name = "MobileNet"
        self.weights_file_aesthetic = "models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5"
        self.weights_file_technical = "models/MobileNet/weights_mobilenet_technical_0.11.hdf5"
        self.nima1 = Nima(self.base_model_name, weights=None)
        self.nima1.build()
        self.nima2 = Nima(self.base_model_name, weights=None)
        self.nima2.build()
        self.nima1.nima_model.load_weights(self.weights_file_aesthetic)
        self.nima2.nima_model.load_weights(self.weights_file_technical)


    def sort_result(self,samples):
        samples_sorted = sorted(samples, key=lambda kv:kv['mean_score_prediction'])
        return samples_sorted

    def image_file_to_json(self,img_path):
        img_dir = os.path.dirname(img_path)
        img_id = os.path.basename(img_path).split('.')[0]

        return img_dir, [{'image_id': img_id}]


    def image_dir_to_json(self,img_dir, img_type='jpg'):
        img_paths = glob.glob(os.path.join(img_dir, '*.'+img_type))

        samples = []
        for img_path in img_paths:
            img_id = os.path.basename(img_path).split('.')[0]
            samples.append({'image_id': img_id})

        return samples


    def predict(self,model, data_generator):
        return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)


    def assessment(self, image_source, img_format='jpg'):

        # Test one image by HST
        img_arr = utils.load_image(image_source, (224,224))
        X=np.array([img_arr])
        X = self.nima1.base_module.preprocess_input(X)
        result = {}

        predictions = self.nima1.nima_model.predict(X)
        result['Diem Tham my'] = round(calc_mean_score(predictions[0]) * 10.0/8.0, 1) * 10

        predictions = self.nima2.nima_model.predict(X)
        result['Diem Ky thuat'] = round(calc_mean_score(predictions[0]) * 10.0/7.5, 1) * 10

        return result



