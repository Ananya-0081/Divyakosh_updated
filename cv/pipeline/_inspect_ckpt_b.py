import torch
ckpt_b = torch.load(r"models/mobilenet_v3_clean.pt", map_location="cpu", weights_only=False)
if isinstance(ckpt_b, dict):
    print("Type: dict, keys:", list(ckpt_b.keys())[:10])
else:
    print("Type: raw state_dict")

