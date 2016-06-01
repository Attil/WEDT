from nltk.cluster import KMeansClusterer as nltk_KMeansClusterer, euclidean_distance

from unsupervised import Clusterer

class KMeansClusterer(Clusterer):

    def train(self, num_of_clusters=7):
        self.clusterer = nltk_KMeansClusterer(num_of_clusters, euclidean_distance)
        vectors = [self.to_vector_space(line) for line in self.lines if line]
        self.clusterer.cluster(vectors)

    def classify(self, sample):
        return self.clusterer.classify(self.to_vector_space(sample))
