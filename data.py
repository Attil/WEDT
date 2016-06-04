from nltk.cluster import KMeansClusterer as nltk_KMeansClusterer
from tree import Tree

from unsupervised import Clusterer

class DataClusterer(Clusterer):
    def train(self, num_of_clusters=7, repeats=4):
        self.vectors = [self.to_vector_space(line) for line in self.lines if line]

        self.tree = Tree()
        self.tree.words = self.vectors

        for i in range(num_of_clusters-1):
            biggest = self.tree.get_biggest()

            biggest.clusterer = nltk_KMeansClusterer(2, self.distance, repeats)
            biggest.clusterer.cluster(biggest.words)

            l = [vector for vector in biggest.words if biggest.clusterer.classify(vector) == 0]
            r = [vector for vector in biggest.words if biggest.clusterer.classify(vector) == 1]

            biggest.words = None
            biggest.left.words = l
            biggest.right.words = r

    def classify(self, sample):
        ret = ''
        tree = self.tree
        while tree:
            if tree.clusterer.classify(self.to_vector_space(sample)) == 0:
                ret += 'l'
                tree = tree.left
            else:
                ret += 'r'
                tree = tree.right

            if tree.words:  # child
                tree = None
        return ret
