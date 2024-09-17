from abc import ABC, abstractmethod

from .constants import ANSWER_KEY, CONTEXT_KEY, QUESTION_KEY


class Tokenizer(ABC):
    pad_ind: int

    def __init__(self):
        pass

    @abstractmethod
    def encode(self, text, **kwargs):
        pass

    @abstractmethod
    def decode(self, inds, **kwargs):
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    def get_state(self):
        return None

    def set_state(self, state):
        pass


class SimpleTokenizer(Tokenizer):
    def __init__(self):
        self.word2ind = {}
        self.ind2word = []
        self._allow_add_word = True
        self.add_word("<pad>")
        self.pad_ind = self.word2ind["<pad>"]
        self.add_word("<sta>")
        self.sta_ind = self.word2ind["<sta>"]

    def add_word(self, word):
        assert self._allow_add_word
        assert word not in self.word2ind
        self.word2ind[word] = len(self.ind2word)
        self.ind2word.append(word)

    def encode(self, text, **kwargs):
        text = text.strip()
        text = text.lower()
        text = text.replace(".", " .")
        text = text.replace(",", " ,")
        text = text.replace("?", " ?")

        words = text.split()
        inds = []
        for w in words:
            if w not in self.word2ind:
                self.add_word(w)
            inds.append(self.word2ind[w])
        return inds

    def decode(self, inds, **kwargs):
        words = [self.ind2word[i] for i in inds]
        return words

    @property
    def vocab_size(self):
        return len(self.ind2word)

    def build_vocab(self, train_data, val_data, test_data):
        for data in [train_data, val_data, test_data]:
            if data is None:
                continue
            for sample in data:
                if CONTEXT_KEY in sample:
                    self.encode(sample[CONTEXT_KEY])
                if QUESTION_KEY in sample:
                    self.encode(sample[QUESTION_KEY])
                if ANSWER_KEY in sample:
                    self.encode(sample[ANSWER_KEY])
        self._allow_add_word = False

    def get_state(self):
        state = {}
        state["word2ind"] = self.word2ind
        state["ind2word"] = self.ind2word
        return state

    def set_state(self, state):
        self.word2ind = state["word2ind"]
        self.ind2word = state["ind2word"]


def get_tokenizer(cfg):
    return SimpleTokenizer()
