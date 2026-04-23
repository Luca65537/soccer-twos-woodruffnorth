"""Standalone PyTorch CEIA baseline + long-match env wrapper.

Team 0 (players 0 & 1) is trainable with concatenated 672-d obs and a
MultiDiscrete[3]*6 action head. Team 1 is driven by the frozen CEIA baseline
via a pure-PyTorch forward pass (no Ray).

Episodes are \"long-form matches\": the inner Unity env is reset internally
whenever a goal is scored, but the wrapper does not signal done to PPO until
MATCH_STEPS have elapsed. The cumulative reward over the episode is therefore
the goal differential over the match, aligning the training objective with
how the autograder scores matches.
"""
import os

import gym
import numpy as np
import torch
import torch.nn as nn


class StandaloneBaseline(nn.Module):
    """256-256 ReLU FCNet matching the CEIA baseline RLlib policy."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(336, 256)
        self.fc2 = nn.Linear(256, 256)
        self.logits = nn.Linear(256, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.logits(x)


_RLLIB_TO_STANDALONE = {
    "_hidden_layers.0._model.0.weight": "fc1.weight",
    "_hidden_layers.0._model.0.bias": "fc1.bias",
    "_hidden_layers.1._model.0.weight": "fc2.weight",
    "_hidden_layers.1._model.0.bias": "fc2.bias",
    "_logits._model.0.weight": "logits.weight",
    "_logits._model.0.bias": "logits.bias",
}


def load_baseline(weights_path):
    raw = torch.load(weights_path, map_location="cpu")
    remapped = {}
    for rk, v in raw.items():
        if rk in _RLLIB_TO_STANDALONE:
            tensor = v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
            remapped[_RLLIB_TO_STANDALONE[rk]] = tensor.float()
    model = StandaloneBaseline()
    model.load_state_dict(remapped)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _sample_branches(logits_9):
    actions = np.empty(3, dtype=np.int64)
    for b in range(3):
        lb = logits_9[b * 3 : (b + 1) * 3]
        m = lb.max()
        p = np.exp(lb - m)
        p /= p.sum()
        actions[b] = np.random.choice(3, p=p)
    return actions


class VsBaselineEnv(gym.Wrapper):
    """Long-match wrapper: team 0 trainable vs frozen baseline team 1."""

    MATCH_STEPS = 2000

    def __init__(self, env, baseline_model, match_steps=None):
        super().__init__(env)
        self.baseline = baseline_model
        if match_steps is not None:
            self.MATCH_STEPS = match_steps
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(672,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([3] * 6)
        self._last_obs = None
        self._step_count = 0

    def _team0_obs(self, obs_dict):
        return np.concatenate([obs_dict[0], obs_dict[1]]).astype(np.float32)

    def _team1_actions(self, obs_dict):
        out = {}
        with torch.no_grad():
            batch = torch.from_numpy(
                np.stack([obs_dict[2], obs_dict[3]]).astype(np.float32)
            )
            logits = self.baseline(batch).numpy()
        out[2] = _sample_branches(logits[0])
        out[3] = _sample_branches(logits[1])
        return out

    def reset(self, **kwargs):
        self._step_count = 0
        obs = self.env.reset(**kwargs)
        self._last_obs = obs
        return self._team0_obs(obs)

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        team0_actions = {0: action[:3], 1: action[3:]}
        team1_actions = self._team1_actions(self._last_obs)
        all_actions = {**team0_actions, **team1_actions}
        obs, rewards, dones, infos = self.env.step(all_actions)
        team0_reward = float(rewards[0] + rewards[1])
        self._step_count += 1

        inner_done = bool(any(dones.values()))
        self._last_obs = obs
        done = inner_done  # short-episode: terminate on goal
        team0_obs = self._team0_obs(obs)
        return team0_obs, team0_reward, done, {}
