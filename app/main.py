from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel

app = FastAPI(
    title="SPS GenAI API",
    description="Bigram text generation and spaCy word embeddings.",
    version="0.1.0",
)

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()


class TextGenerationRequest(BaseModel):
    start_word: str = Field(..., examples=["the"])
    length: int = Field(10, ge=1, le=200)


class EmbeddingRequest(BaseModel):
    word: str = Field(..., examples=["apple"])
    top_n: int = Field(0, ge=0, le=50, description="If > 0, also return nearest neighbours.")


class EmbeddingResponse(BaseModel):
    word: str
    model: str
    dimensions: int
    has_vector: bool
    embedding: list[float]
    neighbors: list[dict] | None = None


class SimilarityRequest(BaseModel):
    word1: str = Field(..., examples=["apple"])
    word2: str = Field(..., examples=["car"])


class MostSimilarRequest(BaseModel):
    word: str = Field(..., examples=["apple"])
    top_n: int = Field(10, ge=1, le=50)


class AnalogyRequest(BaseModel):
    positive: list[str] = Field(..., examples=[["king", "woman"]])
    negative: list[str] = Field(default_factory=list, examples=[["man"]])
    top_n: int = Field(5, ge=1, le=50)


@app.get("/")
def read_root():
    return {
        "Hello": "World",
        "service": "sps_genai",
        "embedding_model": embedding_model.model_name,
        "embedding_dimensions": embedding_model.dimensions,
        "bigram_vocab_size": len(bigram_model),
    }


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding", response_model=EmbeddingResponse)
def get_embedding(request: EmbeddingRequest):
    """Return the 300-dimensional embedding for the query word."""
    word = request.word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="`word` must not be empty.")

    vector = embedding_model.get_embedding(word)
    has_vector = embedding_model.has_vector(word)

    # An out-of-vocabulary word gets a zero vector, which has no neighbours.
    neighbors = None
    if request.top_n > 0 and has_vector:
        neighbors = [
            {"word": w, "similarity": round(s, 4)}
            for w, s in embedding_model.most_similar(word, top_n=request.top_n)
        ]

    return EmbeddingResponse(
        word=word,
        model=embedding_model.model_name,
        dimensions=len(vector),
        has_vector=has_vector,
        embedding=[float(x) for x in vector],
        neighbors=neighbors,
    )


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    for name, value in (("word1", request.word1), ("word2", request.word2)):
        if not value.strip():
            raise HTTPException(status_code=400, detail=f"`{name}` must not be empty.")

    score = embedding_model.similarity(request.word1, request.word2)
    return {
        "word1": request.word1,
        "word2": request.word2,
        "similarity": round(score, 4),
        "word1_has_vector": embedding_model.has_vector(request.word1),
        "word2_has_vector": embedding_model.has_vector(request.word2),
    }


@app.post("/most_similar")
def get_most_similar(request: MostSimilarRequest):
    if not embedding_model.has_vector(request.word):
        raise HTTPException(
            status_code=404,
            detail=f"No vector available for '{request.word}'.",
        )
    results = embedding_model.most_similar(request.word, top_n=request.top_n)
    return {
        "word": request.word,
        "most_similar": [{"word": w, "similarity": round(s, 4)} for w, s in results],
    }


@app.post("/analogy")
def solve_analogy(request: AnalogyRequest):
    if not request.positive:
        raise HTTPException(status_code=400, detail="`positive` must not be empty.")

    _, results = embedding_model.analogy(
        positive=request.positive,
        negative=request.negative,
        top_n=request.top_n,
    )
    expression = " + ".join(request.positive)
    if request.negative:
        expression += " - " + " - ".join(request.negative)

    return {
        "expression": expression,
        "results": [{"word": w, "similarity": round(s, 4)} for w, s in results],
    }
