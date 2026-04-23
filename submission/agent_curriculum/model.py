import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Standalone policy network matching RLlib's FullyConnectedNetwork (FCNet).
    Architecture: obs(672) -> FC(512) -> FC(512) -> logits(18)
    The 18 logits correspond to 6 action branches x 3 choices each.
    Trained with PPO in team_vs_policy mode (single-agent controlling both teammates).
    """

    def __init__(self, obs_size=672, hidden_size=512, num_logits=18):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, num_logits)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.logits(x)
