from numpy import array

class Clusterer:
    def __init__(self, lines):
        self.lines = list(lines)
        self.words = self.get_words()

    def get_words(self):
        ret = set()

        for line in self.lines:
            for word in line:
                ret.add(word)

        return list(ret)


    def to_vector_space(self, line):
        return array([word in line for word in self.words])
