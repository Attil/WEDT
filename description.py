from collections import defaultdict

from numpy import array
from suffix_trees.STree import STree

from data import DataClusterer

class DescriptionClusterer(DataClusterer):
    def train(self, num_of_clusters=7, **kwargs):
        self.suffix_tree = STree([' '.join(line) for line in self.lines])

        super().train(num_of_clusters, **kwargs)

    def to_vector_space(self, line):
        presence = defaultdict(int)
        for word in line:
            results = self.suffix_tree.find_all(word)
            for result in results:
                presence[result] += 1

        return array([presence[i] for i in range(len(self.suffix_tree.word))])
