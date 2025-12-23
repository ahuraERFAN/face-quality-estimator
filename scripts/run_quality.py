import argparse
import os
import torch
import cv2
import pandas as pd

from face_quality.preprocessing import preprocess_face_bgr
from face_quality.quality import l2_quality, sigmoid_mapping
from face_quality.model_wrapper import FaceEmbeddingModel


def run(image_root, model, device):
    wrapper = FaceEmbeddingModel(model, device)
    results = []

    for pid in sorted(os.listdir(image_root)):
        person_dir = os.path.join(image_root, pid)
        if not os.path.isdir(person_dir):
            continue

        for img_name in sorted(os.listdir(person_dir)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            tensor = preprocess_face_bgr(img, device)
            embedding = wrapper.extract(tensor)

            q = l2_quality(embedding)
            Q = sigmoid_mapping(q)

            results.append({
                "image": f"{pid}/{img_name}",
                "q_norm": round(q, 4),
                "Q": Q
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Face Image Quality Estimation Pipeline"
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Path to root directory of face images"
    )
    parser.add_argument(
        "--output",
        default="face_quality_scores.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device: cuda or cpu"
    )

    args = parser.parse_args()

    # Placeholder model (user must replace)
    raise RuntimeError(
        "Please provide your own face embedding model. "
        "See README for details."
    )


if __name__ == "__main__":
    main()
