from sys import argv

from dataloader import DataLoader
from stemmer import Stemmer
from kmeans import KMeansClusterer

def main(args):
    dl = DataLoader()
    stem = Stemmer('porter')

    # files is a list of files, which are lists of lines, which are lists of words
    files = [{element[0]: stem.stem(element[1]) for element in dl.load_data(file) if stem.stem(element[1])} for file in args]

    for file in files:
        file = {k: list(v) for k, v in file.items()}
        kmeans = KMeansClusterer(list(file.values()))
        kmeans.train()
        classified = {}
        for k, v in file.items():
            print('{}: {}'.format(kmeans.classify(v), k))

if __name__ == '__main__':
    main(argv[1:])
