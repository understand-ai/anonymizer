import numpy as np
from PIL import Image

from anonymizer.utils import Box
from anonymizer.anonymization import Anonymizer


def load_np_image(image_path):
    image = Image.open(image_path).convert('RGB')
    np_image = np.array(image)
    return np_image


class MockObfuscator:
    def obfuscate(self, image, boxes):
        obfuscated_image = np.copy(image)
        for box in boxes:
            obfuscated_image[int(box.y_min):int(box.y_max), int(box.x_min):int(box.x_max), :] = 0.0
        return obfuscated_image


class MockDetector:
    def __init__(self, detected_boxes):
        self.detected_boxes = detected_boxes

    def detect(self, image, detection_threshold):
        return self.detected_boxes


class TestAnonymizer:
    @staticmethod
    def test_it_anonymizes_a_single_image():
        np.random.seed(42)  # to avoid flaky tests
        input_image = np.random.rand(128, 64, 3)  # height, width, channels
        obfuscator = MockObfuscator()
        mock_detector = MockDetector([Box(y_min=0, x_min=10, y_max=20, x_max=30, score=0.5, kind=''),
                                      Box(y_min=100, x_min=10, y_max=120, x_max=30, score=0.9, kind='')])
        expected_anonymized_image = np.copy(input_image)
        expected_anonymized_image[0:20, 10:30] = 0.0
        expected_anonymized_image[100:120, 10:30] = 0.0

        anonymizer = Anonymizer(detectors={'face': mock_detector}, obfuscator=obfuscator)
        anonymized_image, detected_boxes = anonymizer.anonymize_image(input_image, detection_thresholds={'face': 0.1})

        assert np.all(np.isclose(expected_anonymized_image, anonymized_image))
        assert detected_boxes == [Box(y_min=0, x_min=10, y_max=20, x_max=30, score=0.5, kind=''),
                                  Box(y_min=100, x_min=10, y_max=120, x_max=30, score=0.9, kind='')]

    @staticmethod
    def test_it_anonymizes_multiple_images(tmp_path):
        np.random.seed(42)  # to avoid flaky tests
        input_images = [np.random.rand(128, 64, 3), np.random.rand(128, 64, 3), np.random.rand(128, 64, 3)]
        obfuscator = MockObfuscator()
        mock_detector = MockDetector([Box(y_min=0, x_min=10, y_max=20, x_max=30, score=0.5, kind=''),
                                      Box(y_min=100, x_min=10, y_max=120, x_max=30, score=0.9, kind='')])
        expected_anonymized_images = list(map(np.copy, input_images))
        for i, _ in enumerate(expected_anonymized_images):
            expected_anonymized_images[i] = (expected_anonymized_images[i] * 255).astype(np.uint8)
            expected_anonymized_images[i][0:20, 10:30] = 0
            expected_anonymized_images[i][100:120, 10:30] = 0
        # write input images to disk
        input_path = tmp_path / 'input'
        input_path.mkdir()
        output_path = tmp_path / 'output'
        for i, input_image in enumerate(input_images):
            image_path = input_path / f'{i}.png'
            pil_image = Image.fromarray((input_image * 255).astype(np.uint8), mode='RGB')
            pil_image.save(image_path)

        anonymizer = Anonymizer(detectors={'face': mock_detector}, obfuscator=obfuscator)
        anonymizer.anonymize_images(str(input_path), output_path=str(output_path), detection_thresholds={'face': 0.1},
                                    file_types=['jpg', 'png'], write_json=False)

        anonymized_images = []
        for image_path in sorted(output_path.glob('**/*.png')):
            anonymized_images.append(load_np_image(image_path))

        for i, expected_anonymized_image in enumerate(expected_anonymized_images):
            assert np.all(np.isclose(expected_anonymized_image, anonymized_images[i]))
