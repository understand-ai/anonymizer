import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from anonymizer.detection.detector import Detector
from anonymizer.obfuscation.obfuscator import Obfuscator


def save_detections(detections, detections_path):
    json_output = []
    for box in detections:
        json_output.append({
            'y_min': box.y_min,
            'x_min': box.x_min,
            'y_max': box.y_max,
            'x_max': box.x_max,
            'score': box.score,
            'kind': box.kind
        })
    with open(detections_path, 'w') as output_file:
        json.dump(json_output, output_file, indent=2)


class Anonymizer:
    def __init__(self, detectors: Dict[str, Detector], obfuscator: Obfuscator):
        self.detectors = detectors
        self.obfuscator = obfuscator

    @staticmethod
    def __save_image(filepath: str, image: Image.Image, include_exif: bool):
        exif = bytes()
        if include_exif and 'exif' in image.info:
            exif = image.info['exif']     
        image.save(filepath, exif=exif)

    def anonymize_image(self, image: Image.Image, detection_thresholds) -> Tuple[Image.Image, Any]:
        np_img = np.array(image)
        
        assert set(self.detectors.keys()) == set(detection_thresholds.keys()),\
            'Detector names must match detection threshold names'
        detected_boxes = []
        for kind, detector in self.detectors.items():
            new_boxes = detector.detect(np_img, detection_threshold=detection_thresholds[kind])
            detected_boxes.extend(new_boxes)
        
        obfuscation = self.obfuscator.obfuscate(np_img, detected_boxes)
        obf_img = Image.fromarray((obfuscation).astype(np.uint8), mode='RGB')
        obf_img.info = image.info
        
        return obf_img, detected_boxes

    def anonymize_images(self, input_path, output_path, detection_thresholds, file_types, write_json: bool, keep_exif: bool):
        print(f'Anonymizing images in {input_path} and saving the anonymized images to {output_path}...')

        Path(output_path).mkdir(exist_ok=True)
        assert Path(output_path).is_dir(), 'Output path must be a directory'

        files = []
        for file_type in file_types:
            files.extend(list(Path(input_path).glob(f'**/*.{file_type}')))

        for input_image_path in tqdm(files):
            # Create output directory
            relative_path = input_image_path.relative_to(input_path)
            (Path(output_path) / relative_path.parent).mkdir(exist_ok=True, parents=True)
            output_image_path = Path(output_path) / relative_path
            output_detections_path = (Path(output_path) / relative_path).with_suffix('.json')

            # Anonymize image
            img = Image.open(str(input_image_path)).convert('RGB')
            anonymized_image, detections = self.anonymize_image(image=img, detection_thresholds=detection_thresholds)
            self.__save_image(str(output_image_path), anonymized_image, keep_exif)
            if write_json:
                save_detections(detections=detections, detections_path=str(output_detections_path))
