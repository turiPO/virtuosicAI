import numpy as np

import nltk
from nltk.lm import MLE
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline


class LyricsMarkovChain(object):
    """A bigram markov chain of lyrics """

    def __init__(self, lyrics, ngram_order=2, sentence_words_amount=6, combinations_threshold=0.000001) -> None:
        self.avg_amount_words_in_lyrics_sentence: int = 10
        # self.vocab = pad_both_ends(nltk.corpus.words.words(), 2)
        self.train, self.vocab = padded_everygram_pipeline(ngram_order, lyrics)
        self.ngram_order = ngram_order
        self.sentence_words_amount = sentence_words_amount
        self.lm = MLE(ngram_order)
        self.lm.fit(self.train, self.vocab)
        self.combinations_threshold = combinations_threshold

    def count_sentence_combinations(self, begin_char="<s>"):
        return self.count_sentence_recursive(begin_char, self.sentence_words_amount)
    
    def get_probabilistic_words(self, context):
        freq = self.lm.counts[[context]]
        summ = self.lm.counts[context]
        for word, occur in freq.items():
            if word == "<s>":
                summ -= occur
                continue
            # freq ordered by amount of occurences
            if occur / summ < self.combinations_threshold:
                break
            yield word
    
    def count_sentence_recursive(self, context, ttl):
        if ttl <= 0 or context.endswith("</s>"):
            return 1
        probabilistic_words = list(self.get_probabilistic_words(context))
        if len(probabilistic_words) == 0:
            return 1
        next_context = context[-self.ngram_order + 1:]
        next_context = next_context + " " if len(next_context) > 0 else next_context
        return sum(self.count_sentence_recursive(next_context + word, ttl - 1) for word in probabilistic_words if word not in context)