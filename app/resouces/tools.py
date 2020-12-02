from time import sleep
import numpy as np
from cv2 import resize

def process_image(img):
    img = resize(img, (224, 224))
    img = img.reshape(1, 224, 224, 3)
    #img = preprocess_input(img)
    return img


def predict(img, model):
    classn = ['bmw_325i_2003', 'bmw_x5_2001', 'chevrolet_impala_2007', 'chevrolet_silverado_2004']
    predictions = model.predict(img)
    label = classn[predictions.argmax()]
    probability_Value = np.amax(predictions)
    if probability_Value > 0.6:
        car_model = label  # mazda_6_2005_22
        car_brand = label.split('_')[0]  # mazda

        result = {
            'brand': car_brand,
            'model': car_model
        }
        return result,probability_Value

    return {
        'brand': 'unknown',
        'model': 'unknown'
    },None
