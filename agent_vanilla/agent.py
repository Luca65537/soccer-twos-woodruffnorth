import os

import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import PolicyNetwork


class TeamAgent(AgentInterface):
    """
    PPO agent trained with team_vs_policy variation (single-agent mode).
    The policy takes concatenated observations from both teammates (672-dim)
    and outputs 6 action branches (3 per player) as a MultiDiscrete action.
    """

    def __init__(self, env):
        super().__init__()
        self.model = PolicyNetwork(obs_size=672, hidden_size=512, num_logits=18)
        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoint.pth"
        )
        if os.path.isfile(weights_path):
            self.model.load_state_dict(
                torch.load(weights_path, map_location="cpu")
            )
        else:
            print("Checkpoint not found at", weights_path)
        self.model.eval()

    def act(self, observation):
        """
        Args:
            observation: dict {player_id: np.array(336,)} for each teammate
        Returns:
            actions: dict {player_id: np.array([a0, a1, a2])} MultiDiscrete([3,3,3])
        """
        player_ids = sorted(observation.keys())
        # Concatenate both players' observations (336 + 336 = 672)
        combined_obs = np.concatenate(
            [observation[pid] for pid in player_ids]
        )
        state = torch.from_numpy(combined_obs).float().unsqueeze(0)

        with torch.no_grad():
            logits = self.model(state).squeeze(0).numpy()

        # 18 logits = 6 branches of 3 choices each
        # Branches 0-2 -> player 0, branches 3-5 -> player 1
        actions = {}
        for i, pid in enumerate(player_ids):
            branch_actions = []
            for b in range(3):
                branch_idx = i * 3 + b
                branch_logits = logits[branch_idx * 3 : (branch_idx + 1) * 3]
                branch_actions.append(np.argmax(branch_logits))
            actions[pid] = np.array(branch_actions)

        return actions
