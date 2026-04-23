"""After TTBB training, extract policy weights into the same standalone
fc1/fc2/logits format used by agent_vanilla/checkpoint.pth, so we can drop
it directly into the submission folder."""
import os
import pickle
import sys

import numpy as np
import torch

REMAP = {
    "_hidden_layers.0._model.0.weight": "fc1.weight",
    "_hidden_layers.0._model.0.bias": "fc1.bias",
    "_hidden_layers.1._model.0.weight": "fc2.weight",
    "_hidden_layers.1._model.0.bias": "fc2.bias",
    "_logits._model.0.weight": "logits.weight",
    "_logits._model.0.bias": "logits.bias",
}


def extract(ckpt_path, out_path):
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    worker_state = pickle.loads(ckpt["worker"])
    policy_state = worker_state["state"]["default_policy"]
    out = {}
    for rk, v in policy_state.items():
        if rk in REMAP:
            t = v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
            out[REMAP[rk]] = t.float()
    assert set(out.keys()) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "logits.weight", "logits.bias"}, f"missing keys: {out.keys()}"
    torch.save(out, out_path)
    print(f"Saved standalone checkpoint ({len(out)} tensors) to {out_path}")
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")


if __name__ == "__main__":
    ckpt = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "checkpoint.pth"
    extract(ckpt, out)
