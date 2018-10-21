import numpy as np

from anonymizer.obfuscation import Obfuscator
from anonymizer.utils import Box


class TestObfuscator:
    @staticmethod
    def test_it_obfuscates_regions():
        obfuscator = Obfuscator()
        np.random.seed(42)  # to avoid flaky tests
        image = np.random.rand(128, 64, 3)  # height, width, channels
        boxes = [Box(y_min=0, x_min=10, y_max=20, x_max=30, score=0, kind=''),
                 Box(y_min=100, x_min=10, y_max=120, x_max=30, score=0, kind='')]

        # copy to make sure the input image does not change
        obfuscated_image = obfuscator.obfuscate(np.copy(image), boxes)

        assert obfuscated_image.shape == (128, 64, 3)
        assert not np.any(np.isclose(obfuscated_image[0:20, 10:30, :], image[0:20, 10:30, :]))
        assert not np.any(np.isclose(obfuscated_image[100:120, 10:30, :], image[100:120, 10:30, :]))
        assert np.all(np.isclose(obfuscated_image[30:90, :, :], image[30:90, :, :]))
