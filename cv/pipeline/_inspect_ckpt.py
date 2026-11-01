import torch
ckpt = torch.load(r"presence/presence_classifier_v1.pt", map_location="cpu", weights_only=False)
print("Keys:", list(ckpt.keys()))

