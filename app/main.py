import base64
import binascii

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.cnn_model import CIFAR10_CLASSES
from app.embedding_model import EmbeddingModel
from app.image_classifier import ImageClassifier

app = FastAPI(
    title="SPS GenAI API",
    description="Bigram text generation, spaCy word embeddings, and CIFAR-10 image classification.",
    version="0.2.0",
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

_image_classifier = None

MAX_IMAGE_BYTES = 10 * 1024 * 1024


def get_image_classifier() -> ImageClassifier:
    """Build the classifier on first use.

    Deferring this keeps the rest of the API usable if the weights file is
    missing, instead of failing the whole application at import time.
    """
    global _image_classifier
    if _image_classifier is None:
        try:
            _image_classifier = ImageClassifier()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="Model weights are not available on this server.",
            ) from exc
    return _image_classifier


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


class Base64ImageRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image bytes.")
    top_k: int = Field(3, ge=1, le=10)


@app.get("/")
def read_root():
    return {
        "Hello": "World",
        "service": "sps_genai",
        "embedding_model": embedding_model.model_name,
        "embedding_dimensions": embedding_model.dimensions,
        "bigram_vocab_size": len(bigram_model),
        "image_classes": len(CIFAR10_CLASSES),
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


@app.get("/classes")
def list_classes():
    """The ten CIFAR-10 classes the image classifier can return."""
    return {"classes": CIFAR10_CLASSES}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...), top_k: int = Form(3)):
    """Classify an uploaded image into one of the ten CIFAR-10 classes."""
    if not 1 <= top_k <= 10:
        raise HTTPException(status_code=400, detail="`top_k` must be between 1 and 10.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")

    classifier = get_image_classifier()
    try:
        result = classifier.predict(contents, top_k=top_k)
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail="File could not be decoded as an image.",
        ) from exc

    return {"filename": file.filename, **result}


@app.post("/classify_base64")
def classify_base64(request: Base64ImageRequest):
    """Same as /classify, for clients that send the image inline as base64."""
    payload = request.image
    # Browsers and many clients send a data URL; keep only the payload.
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")

    try:
        contents = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="`image` is not valid base64.") from exc

    if not contents:
        raise HTTPException(status_code=400, detail="`image` decoded to no data.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")

    classifier = get_image_classifier()
    try:
        return classifier.predict(contents, top_k=request.top_k)
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail="Data could not be decoded as an image.",
        ) from exc
