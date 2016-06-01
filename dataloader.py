from nltk.tokenize import word_tokenize

class DataLoader:
    def load_data(self, filename, language='english'):
        with open(filename, 'r') as f:
            for line in f:
                ret = [word.lower() for word in word_tokenize(line, language)]
                if ret:
                    yield line, ret
