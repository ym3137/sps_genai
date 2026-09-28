"""Bigram model for text generation."""

import random
import re
from collections import Counter, defaultdict


def simple_tokenizer(text: str, frequency_threshold: int | None = None) -> list[str]:
    """Lowercase and split text into word tokens.

    If frequency_threshold is set, tokens rarer than that are dropped. This
    matters on large corpora, where the long tail of rare words produces
    bigrams with meaningless probabilities.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not frequency_threshold:
        return tokens

    word_counts = Counter(tokens)
    return [token for token in tokens if word_counts[token] >= frequency_threshold]


class BigramModel:
    """Estimates p(w2 | w1) = C(w1, w2) / C(w1) and samples text from it."""

    def __init__(self, corpus: list[str], frequency_threshold: int | None = None):
        self.frequency_threshold = frequency_threshold
        text = " ".join(corpus)
        self.vocab, self.bigram_probs = self._analyze_bigrams(text, frequency_threshold)

    @staticmethod
    def _analyze_bigrams(text: str, frequency_threshold: int | None = None):
        words = simple_tokenizer(text, frequency_threshold)
        bigrams = list(zip(words[:-1], words[1:]))

        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]

        return list(unigram_counts.keys()), dict(bigram_probs)

    def generate_text(self, start_word: str, length: int = 20) -> str:
        """Sample `length` words, starting from `start_word`."""
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(max(length - 1, 0)):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:  # no bigram starts with this word, so stop here
                break

            next_word = random.choices(
                list(next_words.keys()),
                weights=list(next_words.values()),
            )[0]
            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)

    def __len__(self) -> int:
        return len(self.vocab)
