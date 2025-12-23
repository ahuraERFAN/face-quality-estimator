import numpy as np

def l2_quality(embedding):
    return np.linalg.norm(embedding)

def sigmoid_mapping(q, mu=0.65, sigma=0.3):
    return round(100 / (1 + np.exp(-(q - mu) / sigma)))
