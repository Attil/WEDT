from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation

class Stemmer:
    def __init__(self, stemmer, language='english'):
        self.language = language
        self.stemmer = PorterStemmer() if stemmer == 'porter' else SnowballStemmer(language)

    def stem(self, line):
        exclude = stopwords.words(self.language)
        return [self.stemmer.stem(word) for word in line if word not in exclude and word not in punctuation]
