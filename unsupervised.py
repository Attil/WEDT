from nltk.cluster import euclidean_distance, cosine_distance
from numpy import array


class Clusterer:
    def __init__(self, lines, distance):
        self.lines = list(lines)
        self.words = self.get_words()
        if distance == 'euclidean':
            self.distance = euclidean_distance
        elif distance == 'cosine':
            self.distance = cosine_distance

    def get_words(self):
        ret = set()

        for line in self.lines:
            for word in line:
                ret.add(word)

        return list(ret)


    def to_vector_space(self, line):
        return array([word in line for word in self.words])
