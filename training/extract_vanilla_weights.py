"""Dump vanilla policy weights (RLlib keys) to a standalone file for warm-start."""
import pickle
import os
import torch
import numpy as np

CKPT = "/home/hice1/lgianantonio3/scratch/soccertwos/soccer-twos-starter/ray_results/vanilla_tvr/PPO_Soccer_19709_00000_0_2026-04-08_23-20-47/checkpoint_000939/checkpoint-939"
OUT = "/home/hice1/lgianantonio3/scratch/soccertwos/soccer-twos-starter/vanilla_rllib_weights.pth"

with open(CKPT, "rb") as f:
    ckpt = pickle.load(f)

worker_state = pickle.loads(ckpt["worker"])
policy_state = worker_state["state"]["default_policy"]

weights = {}
for k, v in policy_state.items():
    if k == "_optimizer_variables":
        continue
    if isinstance(v, np.ndarray):
        weights[k] = torch.from_numpy(v).float()
    elif isinstance(v, torch.Tensor):
        weights[k] = v.float()

print("Extracted keys:")
for k, v in weights.items():
    print(f"  {k}: {tuple(v.shape)}")

torch.save(weights, OUT)
print(f"\nSaved to {OUT}")
