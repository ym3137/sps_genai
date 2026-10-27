# sps_genai

FastAPI service for the Applied Generative AI course.

* Module 3 class activity: bigram text generation
* Assignment 1: spaCy word embeddings
* Assignment 2: a CIFAR-10 image classifier built on the CNN architecture
  specified in the assignment
* Assignment 3: a DCGAN that generates hand-written digits, trained on MNIST

## Project layout

```
sps_genai/
├── app/
│   ├── __init__.py
│   ├── bigram_model.py      # BigramModel
│   ├── embedding_model.py   # EmbeddingModel
│   ├── cnn_model.py         # CNNClassifier and the shared image transform
│   ├── image_classifier.py  # Inference wrapper around the trained weights
│   ├── gan_model.py         # Generator and Discriminator
│   ├── digit_generator.py   # Sampling and PNG encoding
│   ├── main.py              # FastAPI app and endpoints
│   └── weights/
│       ├── cnn_cifar10.pt   # Trained CIFAR-10 weights
│       └── gan_generator.pt # Trained MNIST generator weights
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

On Linux, torch and torchvision resolve from the PyTorch CPU index rather than
PyPI (see `[tool.uv.sources]`). This keeps the Docker image from pulling in
roughly 2.5 GB of CUDA libraries that a CPU container cannot use. macOS wheels
come from PyPI as usual.

The trained CNN and GAN weights are committed to the repository, so nothing needs
to be downloaded or trained before the models work.

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
spaCy model and PyTorch are installed into the image; afterwards the dependency
layer is cached. No model download or training happens at run time.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Health check and model metadata |
| POST | `/generate` | Bigram text generation |
| POST | `/embedding` | Word embedding for a query word |
| POST | `/similarity` | Cosine similarity between two words |
| POST | `/most_similar` | Nearest neighbours in embedding space |
| POST | `/analogy` | Vector arithmetic, e.g. king - man + woman |
| POST | `/classify` | Classify an uploaded image into a CIFAR-10 class |
| POST | `/classify_base64` | Same, for a base64-encoded image |
| GET | `/classes` | List the ten CIFAR-10 class names |
| POST | `/generate_digits` | Generate hand-written digits with the GAN |
| GET | `/generate_digits/preview` | Tiled PNG of generated digits |

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

## Image classification

### `POST /classify`

Takes a multipart file upload and returns the predicted CIFAR-10 class.

```bash
curl -X POST http://127.0.0.1:8000/classify \
     -F "file=@cat.png" \
     -F "top_k=3"
```

```json
{
  "filename": "cat.png",
  "predicted_class": "cat",
  "confidence": 0.8659,
  "top_k": [
    { "class": "cat", "probability": 0.8659 },
    { "class": "dog", "probability": 0.0557 },
    { "class": "ship", "probability": 0.0552 }
  ]
}
```

Any image format Pillow can open works. Images are converted to RGB, resized to
64x64 and normalised with the CIFAR-10 channel statistics, which is the same
`build_transform()` the model was trained with.

`/docs` renders a file picker for this endpoint, which is the quickest way to
try it.

### `POST /classify_base64`

For clients that would rather send the image inline. A `data:image/png;base64,`
prefix is accepted and stripped.

```bash
curl -X POST http://127.0.0.1:8000/classify_base64 \
     -H "Content-Type: application/json" \
     -d "{\"image\": \"$(base64 -w0 cat.png)\", \"top_k\": 3}"
```

## The CNN

`app/cnn_model.py` implements the architecture required by Assignment 2:

| Layer | Configuration | Output shape |
|---|---|---|
| Input | RGB image | 3 x 64 x 64 |
| Conv2D | 16 filters, 3x3, stride 1, padding 1 | 16 x 64 x 64 |
| ReLU | | 16 x 64 x 64 |
| MaxPool2D | 2x2, stride 2 | 16 x 32 x 32 |
| Conv2D | 32 filters, 3x3, stride 1, padding 1 | 32 x 32 x 32 |
| ReLU | | 32 x 32 x 32 |
| MaxPool2D | 2x2, stride 2 | 32 x 16 x 16 |
| Flatten | | 8192 |
| Linear | 100 units | 100 |
| ReLU | | 100 |
| Linear | 10 units | 10 |

825,398 trainable parameters. CIFAR-10 images are 32x32, so they are resized to
64x64 to match the specified input size.

Trained for 20 epochs with Adam at a learning rate of 1e-3, batch size 128, and
random horizontal flips. Test accuracy on the 10,000 held-out CIFAR-10 images is
**69.96%**, from the epoch 9 checkpoint. Accuracy peaks there and then declines
while training accuracy keeps rising, since the specified architecture has no
Batch Normalisation or Dropout to regularise it.

`samples/` holds one test image per class if you want to try the endpoint
straight away:

```bash
curl -X POST http://127.0.0.1:8000/classify -F "file=@samples/truck.png"
```

## Digit generation

### `GET /generate_digits/preview`

The quickest way to see the GAN working. Returns a tiled PNG, so you can open it
straight in a browser:

```
http://127.0.0.1:8000/generate_digits/preview?count=32&seed=7&columns=8
```

### `POST /generate_digits`

Returns the digits as base64-encoded PNGs in JSON.

```bash
curl -X POST http://127.0.0.1:8000/generate_digits \
     -H "Content-Type: application/json" \
     -d '{"count": 16, "seed": 42, "as_grid": true}'
```

```json
{
  "count": 16,
  "seed": 42,
  "latent_dim": 100,
  "format": "png",
  "as_grid": true,
  "images": ["iVBORw0KGgoAAAANS..."]
}
```

`seed` is optional. Setting it makes the request reproducible, so the same call
returns the same digits. `as_grid` tiles them into one image instead of returning
a list.

## The GAN

`app/gan_model.py` implements the architecture required by Assignment 3.

| Generator | Output | | Discriminator | Output |
|---|---|---|---|---|
| Noise input | 100 | | Image input | 1 x 28 x 28 |
| Linear → 7x7x128, reshape | 128 x 7 x 7 | | Conv2d 1→64, k4 s2 p1 | 64 x 14 x 14 |
| ConvT 128→64, k4 s2 p1 | 64 x 14 x 14 | | LeakyReLU(0.2) | 64 x 14 x 14 |
| BatchNorm2d + ReLU | 64 x 14 x 14 | | Conv2d 64→128, k4 s2 p1 | 128 x 7 x 7 |
| ConvT 64→1, k4 s2 p1 | 1 x 28 x 28 | | BatchNorm2d + LeakyReLU(0.2) | 128 x 7 x 7 |
| Tanh | 1 x 28 x 28 | | Flatten + Linear | 1 |

765,761 parameters in the generator and 138,817 in the discriminator.

Trained for 30 epochs on the 60,000 MNIST training images with Adam
(lr 2e-4, betas 0.5/0.999) and batch size 128. The generator uses the
non-saturating loss `-log D(G(z))` rather than `log(1 - D(G(z)))`, which gives a
far stronger gradient while the generator is still poor.

Images are scaled to `[-1, 1]` for training to match the generator's Tanh output,
and scaled back to `[0, 255]` when encoding to PNG.

`samples/gan_digits.png` shows 64 digits from the trained generator. Classifying
5,000 generated samples with a separate MNIST CNN (98.6% test accuracy) finds all
ten digits present, though unevenly: 1 and 4 appear at roughly 40% of their true
frequency while 6 and 7 are over-produced.
