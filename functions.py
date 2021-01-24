from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16 as inc_net
import tensorflow
import numpy as np
import matplotlib.pyplot as plt

print('Modules loaded')

## BEGIGN 0
## MALIGNANT 1
## NORMAL 2

Labels={
    0:'BENIGN',
    1:'MALIGNANT',
    2:'NORMAL'
    }

ABOUT_TEXT='''
Absolute Breast Cancer Detector (ABCD)
Early detection and explanation in detecting breast cancer from USG images.

Tamojit Das      (IEM, Kolkata)
Sayantani Ghosh  (CU, Kolkata)
'''

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

def get_explanation():
    model = tensorflow.keras.models.load_model('Model_1.h5')
    print(1)
    images = transform_img_fn(['test.png'])
    print(2)
    explainer = lime_image.LimeImageExplainer()
    print(3)
    explanation = explainer.explain_instance(images[0].astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=1000)
    print(4)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    print(5)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))


def get_prediction(path):
    model = tensorflow.keras.models.load_model('Model_1.h5')
    print(1)
    images = transform_img_fn([path])
    print(2)
    return Labels[np.argmax(model.predict(images))]

#get_explanation()

#get_prediction()
