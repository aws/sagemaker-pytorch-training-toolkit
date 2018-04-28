import os
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dictionary(object):
    def __init__(self):
        logger.debug('Dictionary __init__')
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        logger.debug('Dictionary add_word')
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        logger.debug('Dictionary __len__')
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        logger.debug('Corpus __init__')
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'wiki.test.tokens'))
        self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))

    def tokenize(self, path):
        logger.debug('Corpus tokenize')
        """Tokenizes a text file."""
        logger.debug('path: {}'.format(path))
        logger.debug('os.path.exists(path): {}'.format(os.path.exists(path)))
        assert os.path.exists(path)
        # Add words to the dictionary
        logger.debug('Add words to the dictionary')
        with open(path, 'rb') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        logger.debug('Tokenize file content')
        with open(path, 'rb') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        logger.debug('len(ids)'.format(len(ids)))
        return ids