import numpy as np
import keras.backend as K
import datetime
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

def loss(y_true, y_pred): return y_pred[0]
def xy_loss(y_true, y_pred): return y_pred[1]
def wh_loss(y_true, y_pred): return y_pred[2]
def confidence_loss(y_true, y_pred): return y_pred[3]
def confidence_loss_obj(y_true, y_pred): return y_pred[4]
def confidence_loss_noobj(y_true, y_pred): return y_pred[5]
def class_loss(y_true, y_pred): return y_pred[6]

gpus = tf.config.experimental.list_physical_devices('GPU')
def _main():
    annotation_path = 'model_data/2007_train.txt'
    log_dir = "logs/008/"
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (416,416) # multiple of 32, hw

    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            model = create_model(input_shape, anchors, num_classes, freeze_body=2, load_pretrained=True,
                    weights_path='logs/007/trained_weights_stage_9.h5') # make sure you know what you freeze
            logging = TensorBoard(log_dir=log_dir)
            checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, verbose=1)
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1)

            # Split validation and train
            val_split = 0.1
            with open(annotation_path) as f:
                lines = f.readlines()
            np.random.seed(10101)
            np.random.shuffle(lines)
            np.random.seed(None)
            num_val = int(len(lines)*val_split)
            num_train = len(lines) - num_val

            # Train with frozen layers first, to get a stable loss.
            # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
            if False:
                model.compile(optimizer=Adam(lr=1e-3), loss={ 'yolo_loss': loss},
                metrics = [xy_loss, wh_loss, confidence_loss, confidence_loss_obj,
                    confidence_loss_noobj, class_loss,])
                batch_size = 8
                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train//batch_size),
                        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                        validation_steps=max(1, num_val//batch_size),
                        epochs=20,
                        initial_epoch=0,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
                model.save_weights(log_dir + 'trained_weights_stage_1.h5')
                model.save('my_model_1.h5')
            # Unfreeze and continue training, to fine-tune.
            # Train longer if the result is not good.
            if True:
                for i in range(len(model.layers)):
                    model.layers[i].trainable = True
                model.compile(optimizer=Adam(lr=1e-4), loss={ 'yolo_loss': loss},
                metrics = [xy_loss, wh_loss, confidence_loss, confidence_loss_obj,
                    confidence_loss_noobj, class_loss,])
                batch_size = 8 # recompile to apply the change
                print('Unfreeze all of the layers.')

                batch_size = 8 # note that more GPU memory is required after unfreezing the body
                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=600,
                    initial_epoch=500,
                    callbacks=[logging, checkpoint])
                model.save_weights(log_dir + 'trained_weights_stage_10.h5')
                model.save('my_model_10.h5')
        except RuntimeError as e:
            print(e)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2, weights_path='model_data/yolo_weights.h5'):
    """create the training model
    Input: input shape of image (416, 416), anchors, number of class, anchors_path
    Output: Keras model
    """
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape # (416, 416)
    num_anchors = len(anchors)
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], num_anchors//3, num_classes+5))
              for l in range(3)]
    # y_true = GT box, shape = (13,13), (26,26), (52,52)
    # y_true = list of array, shape like yolo_outputs, xywh are reletive value
    # y_true = x_min, y_min, width, height, objectness, class_prob

    ### MODEL BODY ###
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    # [y1,y2,y3] = [[num_anchors/3 * (num_classes + ax,ay,aw,ah,ao)][3*25][3*25]]
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    ### LOSS FUNC ###
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """data generator for fit_generator"""
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data) # image np array
        box_data = np.array(box_data) # (x_min, y_min, x_max, y_max, class_id)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size) # TODO: Generator yield

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    "Read line of train.txt and generate the data"
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()