"""
Extract PyTorch weights from Ray RLlib PPO checkpoints into standalone state dicts.
Usage: python extract_checkpoint.py <checkpoint_path> <output_path>

Maps RLlib FCNet weight keys to our standalone PolicyNetwork keys:
  _hidden_layers.0._model.0.{weight,bias} -> fc1.{weight,bias}
  _hidden_layers.1._model.0.{weight,bias} -> fc2.{weight,bias}
  _logits._model.0.{weight,bias}          -> logits.{weight,bias}
"""
import sys
import pickle
import torch
import numpy as np


KEY_MAP = {
    "_hidden_layers.0._model.0.weight": "fc1.weight",
    "_hidden_layers.0._model.0.bias": "fc1.bias",
    "_hidden_layers.1._model.0.weight": "fc2.weight",
    "_hidden_layers.1._model.0.bias": "fc2.bias",
    "_logits._model.0.weight": "logits.weight",
    "_logits._model.0.bias": "logits.bias",
}


def extract(checkpoint_path, output_path):
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    worker_data = pickle.loads(data["worker"])
    policy_state = worker_data["state"]["default_policy"]

    state_dict = {}
    for rllib_key, our_key in KEY_MAP.items():
        tensor = torch.tensor(np.array(policy_state[rllib_key]))
        state_dict[our_key] = tensor
        print(f"  {rllib_key} ({tensor.shape}) -> {our_key}")

    torch.save(state_dict, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_checkpoint.py <checkpoint_path> <output.pth>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
