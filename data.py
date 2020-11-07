import torchtext
from collections import Counter


class Vocab:
    """
    Offers word2index and index2word functionality after counting words in input sentences.
    Allows choosing to only keep the top_k words. Explicitly reserves one index for unknown
    words and assumes custom <sos> and <eos> words are part of sentences during word counting
    with self.add_sentence().
    """
    def __init__(self, sentence_splitter=torchtext.data.utils.get_tokenizer('basic_english')):
        """
        :param sentence_splitter: a function that takes in a string and returns a list
        of words.
        """
        self.counter = Counter()
        self.word2index = dict()
        self.index2word = dict()
        self.splitter = sentence_splitter
        self.UNKNOWN_WORD_INDEX = 0

    def add_sentence(self, sentence):
        """
        Update word counts from sentence after splitting its words.
        :param sentence: a single string.
        """
        self.counter.update(self.splitter(sentence))

    def build_vocab(self, top_k):
        """
        Only keep the top_k words in the vocab.
        :param top_k: how many words to keep
        """
        self.index2word[self.UNKNOWN_WORD_INDEX] = '<unknown>'
        self.word2index['<unknown>'] = self.UNKNOWN_WORD_INDEX

        words = self.counter.most_common(top_k)
        for index, (word, _) in enumerate(words):
            self.word2index[word] = index+1
            self.index2word[index+1] = word

    def word_to_index(self, word):
        try:
            return self.word2index[word]
        except KeyError:
            return self.UNKNOWN_WORD_INDEX

    def index_to_word(self, index):
        try:
            return self.index2word[index]
        except KeyError:
            return self.index2word[self.UNKNOWN_WORD_INDEX]

    def save_vocab(self, root_path):
        """
        Saves the word2index and index2word dictionary in
        a text file 'word2index.txt' at root_path.
        :param root_path: folder at which to save the txt file.
        """
        with open(root_path + 'word2index.txt', 'a') as file:
            for word in self.word2index.keys():
                line = f"{word} {self.word2index[word]}\n"
                file.write(line)

    def load_vocab(self, filepath):
        """
        Initializes the index2word and word2index dictionaries
        from a savefile created by self.save_vocab.
        :param filepath: filepath including filename of txt file
        """
        self.word2index = dict()
        self.index2word = dict()

        with open(filepath) as file:
            for line in file:
                line = line.strip().split()
                word, index = line[0], line[1]

                self.word2index[word] = index
                self.index2word[index] = word
