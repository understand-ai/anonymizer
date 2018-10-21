from anonymizer.utils import Box


class TestBox:
    @staticmethod
    def test_it_has_coordinates_a_score_and_a_kind():
        box = Box(x_min=1.0, y_min=2.0, x_max=3.0, y_max=4.0, score=0.9, kind='face')

        assert box.x_min == 1.0
        assert box.y_min == 2.0
        assert box.x_max == 3.0
        assert box.y_max == 4.0
        assert box.score == 0.9
        assert box.kind == 'face'
