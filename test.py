import torch

state = torch.load("best_asl_model.pth", map_location="cpu")

print(state["classifier.1.weight"].shape)