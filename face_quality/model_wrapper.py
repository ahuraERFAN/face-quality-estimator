import torch

class FaceEmbeddingModel:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def extract(self, tensor):
        embedding = self.model(tensor)
        return embedding.squeeze().cpu().numpy()
