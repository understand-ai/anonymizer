from anonymizer.detection import download_weights


class TestDownloadWeights:
    @staticmethod
    def test_it_downloads_weights(tmp_path):
        weights_directory = tmp_path / 'weights'
        assert len(list(weights_directory.glob('**/*.pb'))) == 0

        download_weights(download_directory=weights_directory, version='1.0.0')

        assert len(list(weights_directory.glob('**/*.pb'))) == 2
        assert (weights_directory / 'weights_face_v1.0.0.pb').is_file()
        assert (weights_directory / 'weights_plate_v1.0.0.pb').is_file()
        assert not (weights_directory / 'nonexistent_path.pb').is_file()
