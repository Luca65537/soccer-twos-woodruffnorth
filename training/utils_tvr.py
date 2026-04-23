"""Utility for team_vs_policy (single-agent) env with reward shaping."""
import gym
import numpy as np
import soccer_twos


class SingleAgentRewardShaper(gym.Wrapper):
    """Reward shaping for single-agent team_vs_policy mode.
    Obs is (672,), reward is scalar, info has nested player dicts."""
    GOAL_X = 10.5
    GAMMA = 0.99
    BALL_GOAL_COEFF = 0.08
    PLAYER_BALL_COEFF = 0.03
    GOAL_REWARD_SCALE = 10.0

    def __init__(self, env):
        super().__init__(env)
        self.prev_pot = None

    def reset(self, **kwargs):
        self.prev_pot = None
        return self.env.reset(**kwargs)

    def _compute_potentials(self, info):
        # In single-agent team_vs_policy, info contains player dicts
        # Try to extract ball/player positions
        ball_pos = None
        player_positions = []
        for key in info:
            if isinstance(info[key], dict) and "ball_info" in info[key]:
                ball_pos = info[key]["ball_info"]["position"]
                player_positions.append(info[key]["player_info"]["position"])

        if ball_pos is None:
            return None

        # Team 0 attacks +x
        ball_goal_pot = -abs(ball_pos[0] - self.GOAL_X)
        dists = [np.sqrt((p[0]-ball_pos[0])**2 + (p[1]-ball_pos[1])**2) for p in player_positions]
        player_ball_pot = -np.mean(dists) if dists else 0

        return {"ball_goal": ball_goal_pot, "player_ball": player_ball_pot}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        base_reward = reward * self.GOAL_REWARD_SCALE
        pots = self._compute_potentials(info)

        if pots is not None and self.prev_pot is not None:
            shaped = (
                base_reward
                + self.BALL_GOAL_COEFF * (self.GAMMA * pots["ball_goal"] - self.prev_pot["ball_goal"])
                + self.PLAYER_BALL_COEFF * (self.GAMMA * pots["player_ball"] - self.prev_pot["player_ball"])
            )
        else:
            shaped = base_reward

        self.prev_pot = pots
        return obs, shaped, done, info


def create_rllib_env_tvr(env_config: dict = {}):
    """Creates a single-agent team_vs_policy env with reward shaping."""
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    env = SingleAgentRewardShaper(env)
    return env
