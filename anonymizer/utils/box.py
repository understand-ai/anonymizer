class Box:
    def __init__(self, x_min, y_min, x_max, y_max, score, kind):
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.score = float(score)
        self.kind = str(kind)

    def __repr__(self):
        return f'Box({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}, {self.score}, {self.kind})'

    def __eq__(self, other):
        if isinstance(other, Box):
            return (self.x_min == other.x_min and self.y_min == other.y_min and
                    self.x_max == other.x_max and self.y_max == other.y_max and
                    self.score == other.score and self.kind == other.kind)
        return False
