# Facial Verification with a Siamese Network

A one-shot facial verification system built with a Siamese neural network in
TensorFlow/Keras, trained on anchor/positive/negative face-image triplets and
verified live via webcam. Based on the approach in Koch et al., *Siamese
Neural Networks for One-shot Image Recognition*.

## How it works

1. **Data collection** — a webcam script captures *anchor* images (reference
   shots of you) and *positive* images (more shots of you). The
   [Labelled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset
   supplies *negative* images (other people).
2. **Preprocessing** — every image is resized to 100x100 and scaled to `[0, 1]`.
   Anchor/positive pairs are labelled `1` (same person), anchor/negative pairs
   are labelled `0` (different person).
3. **Embedding network** — a 4-block CNN maps each 100x100x3 image to a
   4096-dimensional embedding vector.
4. **Siamese model** — two copies of the embedding network (shared weights)
   process an anchor and a candidate image; an `L1Dist` layer computes their
   absolute difference, and a final `Dense(1, sigmoid)` layer outputs a
   similarity score.
5. **Training** — a custom `tf.function` train step with binary cross-entropy
   loss and Adam, with periodic checkpointing.
6. **Verification** — at inference time, a freshly captured frame is compared
   against a folder of reference photos. If enough of them score above a
   detection threshold, the person is "verified".

## Project structure

```
.
├── Faceid.ipynb              # Main notebook: data collection through training & saving
├── layers.py                 # Custom L1Dist layer (needed to save/reload the model)
├── faceid_app.py             # Standalone script for real-time webcam verification
├── requirements.txt
├── data/                     # anchor/ positive/ negative/ image folders (gitignored)
├── application_data/
│   ├── input_image/          # captured frame at verification time (gitignored)
│   └── verification_images/  # your reference photos used for verification (gitignored)
└── training_checkpoints/     # saved during training (gitignored)
```

## Setup

```bash
git clone <your-repo-url>
cd facial-verification-siamese-network
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll need a webcam for the data-collection and real-time verification steps.

## Usage

### 1. Collect your training images

Open `Faceid.ipynb` and run through **Section 1 (Setup)** and **Section 2
(Collect Positives and Anchors)**. Download `lfw.tgz` from the
[LFW website](http://vis-www.cs.umass.edu/lfw/) into the project root first —
it supplies the "negative" (not-you) class.

In the webcam window:
- Press **`a`** to capture an anchor image
- Press **`p`** to capture a positive image
- Press **`q`** to quit

Aim for at least a couple hundred of each for decent results.

### 2. Preprocess, build, and train the model

Run **Sections 3–5** of the notebook. This builds the embedding network and
Siamese model, then trains it (50 epochs by default). Training checkpoints
are saved every 10 epochs to `training_checkpoints/`.

### 3. Evaluate

**Section 6** reports precision/recall on a held-out test split and shows a
sample pair.

### 4. Save the model

**Section 7** saves the trained model to `siamesemodel.h5`.

### 5. Set up your reference photos

Add a handful of clear photos of yourself to
`application_data/verification_images/`.

### 6. Run real-time verification

Either continue in the notebook (**Section 8**) or, once you have
`siamesemodel.h5`, run the standalone app:

```bash
python faceid_app.py
```

- Press **`v`** to capture the current frame and verify it against your
  reference photos
- Press **`q`** to quit

The console prints `VERIFIED` or `NOT VERIFIED` after each check. Tune
`DETECTION_THRESHOLD` and `VERIFICATION_THRESHOLD` at the top of
`faceid_app.py` (or the equivalent cell in the notebook) if you're getting
too many false positives/negatives.

## Notes on this rebuild

This repo was reconstructed from a notebook export after the original source
file was lost. A few things were cleaned up along the way:

- `os.makedirs(..., exist_ok=True)` to avoid `FileExistsError` on re-runs.
- The `L1Dist` custom layer now lives in its own `layers.py` module (rather
  than only in a notebook cell), and its `call()` method accepts a single
  `[a, b]` pair rather than two positional arguments, so it works correctly
  with both the original TF 2.4-era Keras and modern Keras 3.
- `requirements.txt` targets a current, cross-platform TensorFlow release
  instead of the original Windows-specific `tensorflow==2.4.1` /
  `tensorflow-gpu==2.4.1` pinned wheels. If you specifically need the older
  version, install it manually — just note the `L1Dist` fix above is still
  recommended either way.
- Removed leftover debugging cells (typo'd variable names, out-of-order
  cells) that were artifacts of interactive notebook development.

## Background reading

- Koch, Zemel, Salakhutdinov — [*Siamese Neural Networks for One-shot Image Recognition*](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [Labelled Faces in the Wild dataset](http://vis-www.cs.umass.edu/lfw/)

