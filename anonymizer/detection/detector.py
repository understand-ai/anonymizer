import numpy as np
import tensorflow as tf

from anonymizer.utils import Box


class Detector:
    def __init__(self, kind, weights_path):
        self.kind = kind

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(weights_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        conf = tf.ConfigProto()
        self.session = tf.Session(graph=self.detection_graph, config=conf)

    def _convert_boxes(self, num_boxes, scores, boxes, image_height, image_width, detection_threshold):
        assert detection_threshold >= 0.001, 'Threshold can not be too close to "0".'

        result_boxes = []
        for i in range(int(num_boxes)):
            score = float(scores[i])
            if score < detection_threshold:
                continue

            y_min, x_min, y_max, x_max = map(float, boxes[i].tolist())
            box = Box(y_min=y_min * image_height, x_min=x_min * image_width,
                      y_max=y_max * image_height, x_max=x_max * image_width,
                      score=score, kind=self.kind)
            result_boxes.append(box)
        return result_boxes

    def detect(self, image, detection_threshold):
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        image_height, image_width, channels = image.shape
        assert channels == 3, f'Invalid number of channels: {channels}. ' \
                              f'Only images with three color channels are supported.'

        np_images = np.array([image])
        num_boxes, scores, boxes = self.session.run(
            [num_detections, detection_scores, detection_boxes],
            feed_dict={image_tensor: np_images})

        converted_boxes = self._convert_boxes(num_boxes=num_boxes[0], scores=scores[0], boxes=boxes[0],
                                              image_height=image_height, image_width=image_width,
                                              detection_threshold=detection_threshold)
        return converted_boxes
