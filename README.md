
# Face Image Quality Estimation Pipeline

A reusable, model-agnostic pipeline for estimating face image quality using deep face embeddings.  
This project is designed for research, dataset curation, and biometric preprocessing workflows and follows a standardized image processing pipeline.

---

## Overview

This repository provides a **face image quality assessment pipeline** that estimates the intrinsic quality of a face image based on the properties of its deep embedding.

The system:
- Accepts pre-aligned face images
- Applies a standardized preprocessing workflow
- Extracts face embeddings using a user-provided model
- Computes an intrinsic quality metric
- Maps it to a normalized quality score in the range **0–100**

No pretrained models, private datasets, or proprietary resources are included.

---

## Quality Estimation Principle

Let **V ∈ ℝ⁵¹²** be the face embedding produced by a deep face recognition model.

The raw quality score is defined as the L2 norm of the embedding:

```

q = ||V||₂

```

The final quality score is obtained by mapping `q` to a normalized range using a sigmoid function:

```

Q = round(100 × sigmoid(q))

```

Higher embedding norms generally correspond to higher-quality face images.

---

## Processing Pipeline

The image preprocessing pipeline follows a strict and deterministic workflow:

```

BGR Face Image
→ Resize (192 × 192, bilinear interpolation)
→ Crop (112 × 112)
→ Normalize pixel values (/255)
→ Tensor conversion
→ Face embedding model
→ 512-dimensional feature vector
→ L2 norm computation
→ Sigmoid mapping
→ Quality score (0–100)

```

This ensures consistent quality estimation across datasets and models.

---

## Project Structure

```

face-quality-estimator/
│
├── face_quality/
│   ├── preprocessing.py
│   ├── quality.py
│   ├── model_wrapper.py
│   └── **init**.py
│
├── scripts/
│   └── run_quality.py
│
├── requirements.txt
└── README.md

````

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
````

---

## Usage

### 1. Prepare your face images

* Images should be **pre-aligned face crops**
* Supported formats: JPG, PNG, JPEG
* Directory structure example:

```
dataset/
├── person_001/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── person_002/
│   └── ...
```

---

### 2. Provide your own face embedding model

Any deep face recognition model that outputs a **512-dimensional embedding** can be used, such as:

* ArcFace
* MagFace
* iResNet-based models
* Custom-trained face embedding networks

The model must accept a tensor of shape `(1, C, 112, 112)` and return a 512-D vector.

---

### 3. Run quality estimation

Example usage:

```python
from scripts.run_quality import run

df = run(
    image_root="path/to/face_images",
    model=your_model,
    device="cuda"
)
```

The output is a table containing:

* Image path
* Raw embedding norm (`q`)
* Normalized quality score (`Q`, range 0–100)

---

## Applications

This pipeline can be used for:

* Face dataset cleaning and filtering
* Image quality benchmarking
* Biometric preprocessing pipelines
* Face recognition research
* Robustness analysis of face embeddings

---

## Disclaimer

* This repository does **not** distribute pretrained models or private data.
* Users must ensure proper licensing for any face recognition models they use.
* The project is intended for **research and educational purposes** only.
* The authors are not responsible for misuse of the provided code.

---
"# face-quality-estimator" 
