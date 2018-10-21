import numpy as np
from PIL import Image

from anonymizer.utils import Box
from anonymizer.detection import Detector
from anonymizer.detection import download_weights, get_weights_path


def box_covers_box(covering_box: Box, covered_box: Box):
    return (covered_box.x_min > covering_box.x_min and covered_box.y_min > covering_box.y_min and
            covered_box.x_max < covering_box.x_max and covered_box.y_max < covering_box.y_max)


def load_np_image(image_path):
    image = Image.open(image_path).convert('RGB')
    np_image = np.array(image)
    return np_image


class TestDetector:
    @staticmethod
    def test_it_detects_obvious_faces(tmp_path):
        weights_directory = tmp_path / 'weights'
        face_weights_path = get_weights_path(weights_directory, kind='face')
        download_weights(weights_directory)

        detector = Detector(kind='face', weights_path=face_weights_path)
        np_image = load_np_image('./test/detection/face_test_image.jpg')

        left_face = Box(x_min=267, y_min=64, x_max=311, y_max=184, score=0.0, kind='face')
        right_face = Box(x_min=369, y_min=68, x_max=420, y_max=152, score=0.0, kind='face')

        boxes = detector.detect(np_image, detection_threshold=0.2)

        assert len(boxes) >= 2
        for box in boxes:
            assert box.score >= 0.2
        assert boxes[0].score >= 0.5 and boxes[1].score >= 0.5
        assert ((box_covers_box(boxes[0], left_face) and box_covers_box(boxes[1], right_face)) or
                (box_covers_box(boxes[1], left_face) and box_covers_box(boxes[0], right_face)))
