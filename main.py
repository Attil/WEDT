from collections import defaultdict
from sys import argv

from dataloader import DataLoader
from stemmer import Stemmer
from data import DataClusterer
from description import DescriptionClusterer

def test_clusterer(clusterer, lines):
    print('Training')
    clusterer.train(num_of_clusters=32)

    results = defaultdict(list)

    print('Classyfing')
    for k, v in lines.items():
        results[clusterer.classify(v)].append(k)

    print('Results')

    for k, v in results.items():
        print('Category {}'.format(k))
        for result in v:
            print('\t{}'.format(result))

def main(args):
    dl = DataLoader()
    stem = Stemmer('porter')

    # files is a list of files, which are lists of lines, which are lists of words
    files = [{element[0]: stem.stem(element[1]) for element in dl.load_data(file) if stem.stem(element[1])} for file in args]

    for file, arg in zip(files, args):
        print('Processing file {}...'.format(arg))
        file = {k: list(v) for k, v in file.items()}

        print('Data Clusterer')
        test_clusterer(DataClusterer(list(file.values()), 'euclidean'), file)

        print('-'*64)

        print('Description Clusterer')
        test_clusterer(DescriptionClusterer(list(file.values()), 'cosine'), file)

if __name__ == '__main__':
    main(argv[1:])
