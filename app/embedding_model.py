"""Word embeddings using spaCy's en_core_web_lg vectors."""

import functools

import numpy as np
import spacy
from spacy.language import Language

DEFAULT_MODEL = "en_core_web_lg"


@functools.lru_cache(maxsize=2)
def load_model(model_name: str = DEFAULT_MODEL) -> Language:
    """Load a spaCy pipeline, cached so the 382 MB model loads only once."""
    return spacy.load(model_name)


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.nlp = load_model(model_name)

    @property
    def dimensions(self) -> int:
        return self.nlp.meta["vectors"]["width"]

    def get_embedding(self, text: str) -> np.ndarray:
        """Vector for a word. For a phrase spaCy averages the word vectors."""
        return self.nlp(text).vector

    def has_vector(self, text: str) -> bool:
        return bool(self.nlp(text).has_vector)

    def similarity(self, text1: str, text2: str) -> float:
        return float(self.nlp(text1).similarity(self.nlp(text2)))

    def most_similar(self, text: str, top_n: int = 10) -> list[tuple[str, float]]:
        vector = self.get_embedding(text)
        return self._neighbors(vector, top_n=top_n, exclude={text.lower()})

    def analogy(self, positive: list[str], negative: list[str], top_n: int = 5):
        """Add and subtract word vectors, e.g. king - man + woman."""
        composite = np.zeros(self.dimensions, dtype="float32")
        for word in positive:
            composite = composite + self.get_embedding(word)
        for word in negative:
            composite = composite - self.get_embedding(word)

        exclude = {w.lower() for w in (*positive, *negative)}
        return composite, self._neighbors(composite, top_n=top_n, exclude=exclude)

    def _neighbors(self, vector, top_n: int, exclude: set[str]):
        vector = np.asarray(vector, dtype="float32")
        norm = np.linalg.norm(vector)
        if norm == 0:
            return []

        # The vector table holds separate entries for "apple", "Apple" and
        # "APPLE", so ask for extra results and collapse them by lowercase.
        keys, _, scores = self.nlp.vocab.vectors.most_similar(
            (vector / norm).reshape(1, -1),
            n=(top_n + len(exclude) + 1) * 6,
        )

        results = []
        seen = set(exclude)
        for key, score in zip(keys[0], scores[0]):
            word = self.nlp.vocab.strings[key].lower()
            if word in seen:
                continue
            seen.add(word)
            results.append((word, float(score)))
            if len(results) == top_n:
                break
        return results
