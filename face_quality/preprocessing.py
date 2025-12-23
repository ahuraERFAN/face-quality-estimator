import numpy as np
import torch
import torch.nn.functional as F

def preprocess_face_bgr(img_bgr, device):
    tensor = torch.from_numpy(
        img_bgr.astype(np.float32).transpose(2, 0, 1)
    ).unsqueeze(0)

    resized = F.interpolate(
        tensor, size=(192, 192),
        mode="bilinear", align_corners=False
    )

    resized = resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

    crop = resized[33:192-47, 40:192-40]
    crop = crop.astype(np.float32) / 255.0
    crop = crop.transpose(2, 0, 1)

    return torch.from_numpy(crop).unsqueeze(0).to(device)
