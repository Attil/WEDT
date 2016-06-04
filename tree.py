class Tree:
    def __init__(self):
        self.words = None

    def get_biggest(self):
        if self.words:      # leaf
            return self

        left_biggest = self.left.get_biggest()
        right_biggest = self.right.get_biggest()

        return left_biggest if len(left_biggest.words) > len(right_biggest.words) else right_biggest

    @property
    def left(self):
        try:
            return self._left
        except AttributeError:
            self._left = Tree()
            return self._left

    @property
    def right(self):
        try:
            return self._right
        except AttributeError:
            self._right = Tree()
            return self._right
