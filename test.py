# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from yolo3.model import yolo_body, yolo_head
from train import create_model
from yolo import YOLO
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
# load yolov3 model


tf.debugging.set_log_device_placement(True)

try:
    with tf.device('/device:GPU:1'):
        annotation_path = 'model_data/2007_train.txt'
        log_dir = 'logs/000/'
        classes_path = 'model_data/voc_classes.txt'
        anchors_path = 'model_data/yolo_anchors.txt'

        photo_origin_filename = 'person_3'
        photo_filename = './test_data/' + photo_origin_filename + '.jpg'
        print(photo_filename)
        model_name = 'trained_weights_stage_1'
        yolo = YOLO(model_path="./model_data/" + model_name + ".h5",
                classes_path=classes_path, anchors_path=anchors_path)
        image = Image.open(photo_filename)
        image = yolo.detect_image(image)
        result_name = photo_origin_filename + '_' + model_name + '_result.png'
        image.save('./test_data/result_no_preweight/' + result_name)
except RuntimeError as e:
  print(e)
