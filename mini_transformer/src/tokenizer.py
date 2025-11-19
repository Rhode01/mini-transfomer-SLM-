class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def fit(self, sentence_list):
        for sentence_column in sentence_list:
            for sentence in sentence_column:
                for word in sentence.lower().split():
                    if word not in self.word2idx:
                        self.word2idx[word] = self.idx
                        self.idx2word[self.idx] = word
                        self.idx += 1

    def encode(self, sentence):
        return [self.word2idx[word] for word in sentence.lower().split()]

    def decode(self, indices):
        return " ".join([self.idx2word[i] for i in indices])
