
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_body, yolo_eval, yolo_boxes_and_scores
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)

		return self.label

	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]

		return self.score

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "image_path": 'non.jpg',
        "anchors_path": './model_data/yolo_anchors.txt',
        "classes_path": './model_data/voc_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        image_path = os.path.expanduser(self.image_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        print(model_path + "yolo model load")
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

        image = Image.open(image_path)
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        self.input_image_shape = [image.size[1], image.size[0]]

        yhat = self.yolo_model.predict(image_data)
        #print(np.shape(yhat))
        #yhat = K.cast(yhat, 'float32')
        boxes, scores, classes = yolo_eval(yhat, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes


    def detect_image(self):
        start = timer()
        image_path = os.path.expanduser(self.image_path)
        image = Image.open(image_path)
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        return self.boxes, self.scores, self.classes
        print(np.shape(out_boxes), np.shape(out_scores), np.shape(out_classes))

        print(out_boxes[0], out_scores[0], out_classes)

        font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 15)
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            print("inside")
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            print(box)
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image
        exit()
        #yhat = yolo_boxes_and_scores(yhat[], self.anchors, num_classes, K.shape(yhat[0])[1:3] * 32, self.input_image_shape)
        #print(np.shape(yhat[0][0]))
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        total_boxes = []
        for i in range(len(yhat)):

            _boxes, _box_scores = yolo_boxes_and_scores(yhat[i], self.anchors, num_classes, K.shape(yhat[0])[1:3] * 32, self.input_image_shape)

            netout = yhat[i][0]
            grid_h, grid_w = netout.shape[:2]
            nb_box = 3
            netout = netout.reshape((grid_h, grid_w, nb_box, -1))
            nb_class = netout.shape[-1] - 5
            boxes = []
            obj_thresh = 0.5
            anchors = [[10,13, 16,30, 33,23], [116,90, 156,198, 373,326], [30,61, 62,45, 59,119]]
            for i in range(grid_h*grid_w):
                row = i / grid_w
                col = i % grid_w
                for b in range(nb_box):
                    # 4th element is objectness score
                    objectness = netout[int(row)][int(col)][b][4]
                    if(objectness.all() <= obj_thresh): continue
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[int(row)][int(col)][b][:4]
                    print(x,y,w,h)
                    print(anchors[2 * b + 0])
                    x = (col + x) / grid_w # center position, unit: image width
                    y = (row + y) / grid_h # center position, unit: image height
                    w = np.array(anchors[2 * b + 0], np.float32) * np.exp(w) / 416. # unit: image width
                    h = np.array(anchors[2 * b + 1], np.float32) * np.exp(h) / 416. # unit: image height
                    # last elements are class probabilities
                    print(x,y,w,h)
                    classes = netout[int(row)][col][b][5:]
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
                    print(x-w/2, y-h/2, x+w/2, y+h/2)
                    boxes.append(box)
            # decode the output of the network
            total_boxes += boxes


        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                tf.compat.v1.keras.backend.learning_phase(): 0
            })
        #y = self.yolo_model.predict(image_data)
        #self.boxes, self.scores, self.classes = y
        #out_boxes, out_scores, out_classes = self.boxes, self.scores, self.classes
        font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 15)
        thickness = (image.size[0] + image.size[1]) // 300
        out_boxes = [y[0][..., :4], y[1][..., :4], y[2][..., :4]]
        out_scores = [y[0][..., 4], y[1][..., 4], y[2][..., 4]]
        out_classes = [y[0][..., 5], y[1][..., 5], y[2][..., 5]]

        for i, c in reversed(list(enumerate(out_classes))):
            print(i)
            #predicted_class = self.class_names[c]
            box = out_boxes[i]
            print(box)
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image


if __name__ == '__main__':
    _main()