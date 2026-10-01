import torch, sys
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"Step: {ckpt.get('step', 'N/A')}")
print(f"Has optimizer: {'optimizer_state_dict' in ckpt}")
print(f"Keys: {list(ckpt.keys())}")
