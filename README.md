# sps_genai

FastAPI service for the Applied Generative AI course. Started from the Module 3
class activity (bigram text generation) and extended for Assignment 1 with
spaCy word embeddings.

## Project layout

```
sps_genai/
├── app/
│   ├── __init__.py
│   ├── bigram_model.py      # BigramModel
│   ├── embedding_model.py   # EmbeddingModel
│   └── main.py              # FastAPI app and endpoints
├── Dockerfile
├── .dockerignore
├── pyproject.toml
└── uv.lock
```

## Requirements

* Python 3.12+
* [uv](https://docs.astral.sh/uv/)
* Docker (optional)

`en_core_web_lg` is listed as a project dependency in `pyproject.toml`, so
`uv sync` installs the model along with everything else. There is no separate
`spacy download` step.

## Running locally

```bash
cd sps_genai
uv sync
uv run fastapi dev app/main.py
```

The API runs at <http://127.0.0.1:8000>, with Swagger docs at
<http://127.0.0.1:8000/docs>.

## Running with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

The container listens on port 80, and `-p 8000:80` maps it to the same
localhost port as the dev server. The first build takes a while because the
spaCy model is downloaded into the image.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Health check and model metadata |
| POST | `/generate` | Bigram text generation |
| POST | `/embedding` | Word embedding for a query word |
| POST | `/similarity` | Cosine similarity between two words |
| POST | `/most_similar` | Nearest neighbours in embedding space |
| POST | `/analogy` | Vector arithmetic, e.g. king - man + woman |

### `POST /embedding`

Takes a word, returns its 300-dimensional vector.

```bash
curl -X POST http://127.0.0.1:8000/embedding \
     -H "Content-Type: application/json" \
     -d '{"word": "apple", "top_n": 5}'
```

`top_n` defaults to 0. Set it above 0 to also get that many nearest
neighbours.

```json
{
  "word": "apple",
  "model": "en_core_web_lg",
  "dimensions": 300,
  "has_vector": true,
  "embedding": [-0.3639, 0.4377, -0.2045, "..."],
  "neighbors": [
    { "word": "apples", "similarity": 0.7504 },
    { "word": "blackberry", "similarity": 0.6921 }
  ]
}
```

### `POST /similarity`

```bash
curl -X POST http://127.0.0.1:8000/similarity \
     -H "Content-Type: application/json" \
     -d '{"word1": "apple", "word2": "car"}'
```

```json
{ "word1": "apple", "word2": "car", "similarity": 0.2175,
  "word1_has_vector": true, "word2_has_vector": true }
```

### `POST /analogy`

```bash
curl -X POST http://127.0.0.1:8000/analogy \
     -H "Content-Type: application/json" \
     -d '{"positive": ["king", "woman"], "negative": ["man"], "top_n": 3}'
```

Returns `queen` at 0.79, then `prince` and `kings`.

### `POST /generate`

```bash
curl -X POST http://127.0.0.1:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"start_word": "the", "length": 15}'
```

Output varies between calls, since the next word is sampled from the bigram
distribution rather than picked greedily.
